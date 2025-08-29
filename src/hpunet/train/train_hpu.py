from __future__ import annotations
import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from hpunet.utils.config import load_config
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.sched_optim import make_optimizer, make_scheduler
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.geco import GECO, GECOConfig
from hpunet.models.hpu_net import HPUNet
from hpunet.losses.topk_ce import make_recon_loss
from hpunet.utils.logger import Logger
from dataclasses import is_dataclass, asdict


def auto_pos_weight(y_target: torch.Tensor, pad_mask: torch.Tensor, clip: float = 20.0) -> torch.Tensor | None:
    """per-batch positive-class weight for BCE (negatives / positives)"""
    valid = pad_mask.float().unsqueeze(1)  # [B,1,H,W]
    n_pos = (y_target * valid).sum()
    n_valid = valid.sum()
    if n_pos > 0:
        w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, clip)
        return torch.tensor([float(w.item())], device=y_target.device)
    return None


def masked_bce_sum_per_image(
    logits: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor, pos_weight: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    return (sum_i, count_i) where:
      sum_i   = per-image SUM of BCE over valid pixels [B]
      count_i = per-image COUNT of valid pixels (for reporting) [B]
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight).squeeze(1)  # [B,H,W]
    valid = pad_mask.bool()
    sums   = (bce * valid).flatten(1).sum(dim=1)   # [B]
    counts = valid.flatten(1).sum(dim=1)           # [B]
    return sums, counts


def set_seed(seed: int):
    """set random seeds everywhere for reproducibility"""
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=None)  # uses config value if not specified
    ap.add_argument("--outdir", type=Path, default=Path("runs/hpu"))
    ap.add_argument("--save-name", type=str, default="hpu_last.pth")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)
    
    # use config total_steps if max_steps not specified
    max_steps = args.max_steps if args.max_steps is not None else getattr(cfg, 'total_steps', 240000)
    eval_every_steps = getattr(cfg, 'eval_every_steps', 5000)
    ckpt_every_steps = getattr(cfg, 'ckpt_every_steps', 10000)

    data_root = args.data_root or (args.project_root / "data" / "lidc_crops")
    train_csv = data_root / "train.csv"

    # setup data loader
    train_ds = LIDCCropsDataset(
        csv_path=train_csv, project_root=args.project_root,
        image_size=128, augment=cfg.augment, seed=cfg.seed
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # model with correct params and weight init
    model = HPUNet(in_ch=1, base=24, z_ch=1).to(device)  # z_ch=1 for scalar latents

    # GECO setup with paper defaults
    gcfg = cfg.geco if hasattr(cfg, 'geco') and cfg.geco else {
        "kappa": 0.05,         # LIDC paper spec
        "alpha": 0.99,         # EMA decay
        "lambda_init": 1.0,    # initial multiplier
        "step_size": 0.01      # paper spec
    }
    geco = GECO(GECOConfig(**gcfg)).to(device)

    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    use_topk = bool(getattr(cfg, "use_topk", True))
    k_frac   = float(getattr(cfg, "k_frac", 0.02))
    posw_mode = getattr(cfg, "pos_weight", "auto")
    posw_clip = float(getattr(cfg, "pos_weight_clip", 20.0))
    recon_strategy = getattr(cfg, "recon_strategy", "random")

    # prep config for saving (used in checkpoints)
    if is_dataclass(cfg):
        cfg_to_save = asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_to_save = cfg
    else:
        cfg_to_save = vars(cfg)

    running = {"L": 0.0, "recon_pp": 0.0, "recon_sum": 0.0, "kl": 0.0, "lam": 0.0, "C": 0.0, "Cbar": 0.0}
    logger = Logger(args.outdir, use_tensorboard=True)

    model.train()
    data_iter = iter(train_loader)
    
    for step in range(1, int(max_steps) + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x = batch["image"].to(device, non_blocking=True)        # [B,1,H,W]
        y_all = batch["masks"].to(device, non_blocking=True)    # [B,4,H,W]
        pm = batch["pad_mask"].to(device, non_blocking=True)    # [B,H,W]

        # choose target grader
        y_target, target_info = select_targets_from_graders(y_all, strategy=recon_strategy)

        # forward with posterior sampling (training mode)
        logits, info = model(x, y_target=y_target, sample_posterior=True)
        
        # get KL term
        kl = info.get("KL_sum", torch.tensor(0.0, device=device))
        if kl.ndim > 0:
            kl = kl.mean()

        # handle class balancing
        pos_weight = None
        if posw_mode == "auto":
            pos_weight = auto_pos_weight(y_target, pm, clip=posw_clip)
        elif isinstance(posw_mode, (int, float)):
            pos_weight = torch.tensor([float(posw_mode)], device=device)

        # reconstruction loss computation
        if use_topk:
            # use stochastic top-k as per paper
            recon_loss_pp = make_recon_loss(
                logits=logits, 
                y_target=y_target, 
                pad_mask=pm,
                use_topk=True, 
                k_frac=k_frac,
                stochastic_topk=True  # use Gumbel-Softmax sampling
            )
            # convert per-pixel loss to sum form for GECO
            total_valid_pixels = pm.sum().clamp_min(1).float()
            recon_sum_batch = recon_loss_pp * total_valid_pixels  # total sum over batch
            recon_sum_mean = recon_sum_batch / cfg.batch_size     # mean sum per image
            valid_pix_mean = total_valid_pixels / cfg.batch_size  # mean valid pixels per image
            recon_pp = recon_loss_pp.detach()                     # per-pixel for GECO update
        else:
            # standard BCE loss (sum form)
            sum_i, cnt_i = masked_bce_sum_per_image(logits, y_target, pm, pos_weight=pos_weight)
            recon_sum_mean = sum_i.mean()                         # mean over images (sum-form)
            valid_pix_mean = cnt_i.float().mean().clamp_min(1.0)  # mean valid pixels
            recon_pp = (sum_i.sum() / cnt_i.clamp_min(1).sum()).detach()  # per-pixel (for GECO)

        # GECO update (uses per-pixel reconstruction)
        ge = geco.step(recon_pp)
        lam = torch.tensor(float(ge["lambda"]), device=device)

        # lagrangian using correct GECO kappa
        kappa_pp = geco.cfg.kappa  # access kappa from GECO config
        loss = kl + lam * (recon_sum_mean - kappa_pp * valid_pix_mean)

        # backprop step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # logging stuff
        lr = optimizer.param_groups[0]["lr"]
        logger.log_scalars(step, {
            "train/L": float(loss),
            "train/recon_pp": float(recon_pp),
            "train/recon_sum_mean": float(recon_sum_mean),
            "train/kl": float(kl),
            "train/lambda": float(ge["lambda"]),
            "train/C": float(ge["C"]),
            "train/C_bar": float(ge["C_bar"]),
            "train/lr": lr,
            "train/valid_pix_mean": float(valid_pix_mean),
        })

        # running averages for console output
        running["L"]       += float(loss)
        running["recon_pp"]+= float(recon_pp)
        running["recon_sum"]+= float(recon_sum_mean)
        running["kl"]      += float(kl)
        running["lam"]     += float(ge["lambda"])
        running["C"]       += float(ge["C"])
        running["Cbar"]    += float(ge["C_bar"])

        if step % 100 == 0:
            n = 100.0
            print(
                f"[step {step:5d}] L={running['L']/n:.4f}  "
                f"recon_pp={running['recon_pp']/n:.4f}  recon_sum={running['recon_sum']/n:.4f}  "
                f"kl={running['kl']/n:.4f}  lam={running['lam']/n:.3f}  "
                f"C={running['C']/n:.4f}  CÌ„={running['Cbar']/n:.4f}  lr={lr:.6g}"
            )
            running = {"L": 0.0, "recon_pp": 0.0, "recon_sum": 0.0, "kl": 0.0, "lam": 0.0, "C": 0.0, "Cbar": 0.0}

        # save checkpoints periodically  
        if step % ckpt_every_steps == 0:
            ckpt_path = args.outdir / f"hpu_step_{step}.pth"
            torch.save({
                "step": step, 
                "model": model.state_dict(), 
                "geco": {  # manual GECO state extraction
                    "log_lambda": geco.log_lambda,
                    "ema_c": geco.ema_c,
                    "cfg": asdict(geco.cfg)
                },
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg_to_save
            }, ckpt_path)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")

        # eval placeholder (could add validation here)
        if step % eval_every_steps == 0:
            print(f"[step {step:5d}] Evaluation placeholder (add validation logic here)")
            # TODO: add validation loop here if you have validation data

    # save final checkpoint
    ckpt_path = args.outdir / args.save_name
    torch.save({
        "step": int(max_steps), 
        "model": model.state_dict(), 
        "geco": {
            "log_lambda": geco.log_lambda,
            "ema_c": geco.ema_c, 
            "cfg": asdict(geco.cfg)
        },
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg_to_save
    }, ckpt_path)
    print(f"Final checkpoint saved to: {ckpt_path}")
    logger.close()


if __name__ == "__main__":
    main()