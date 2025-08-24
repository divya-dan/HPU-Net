from __future__ import annotations
import argparse, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch import nn

from hpunet.utils.config import load_config
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.sched_optim import make_optimizer, make_scheduler
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.topk_ce import masked_bce_with_logits, masked_topk_bce_with_logits
from hpunet.models.spu_net import ProbUNet
from hpunet.utils.logger import Logger


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--beta", type=float, default=1.0, help="ELBO beta for KL term")
    ap.add_argument("--outdir", type=Path, default=Path("runs/spu"))
    ap.add_argument("--save-name", type=str, default="spu_last.pth")

    args = ap.parse_args()

    cfg = load_config(args.config)
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_root = args.data_root or (args.project_root / "data" / "lidc_crops")
    train_csv = data_root / "train.csv"

    # --- data
    train_ds = LIDCCropsDataset(
        csv_path=train_csv, project_root=args.project_root,
        image_size=128, augment=cfg.augment, seed=cfg.seed
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # --- model / opt / sched
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    logger = Logger(args.outdir, use_tensorboard=True)
    t0 = time.time()

    model.train()
    data_iter = iter(train_loader)
    for step in range(1, int(args.max_steps) + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x = batch["image"].to(device, non_blocking=True)        # [B,1,H,W]
        y_all = batch["masks"].to(device, non_blocking=True)    # [B,4,H,W]
        pm = batch["pad_mask"].to(device, non_blocking=True)    # [B,H,W]

        # choose target grader mask
        y_target, info_sel = select_targets_from_graders(y_all, strategy=cfg.recon_strategy)

        # forward with posterior sampling q(z|x,y)
        logits, info = model(x, y_target=y_target, sample_posterior=True)

        # Build positive-class weight
        pos_weight = None
        if getattr(cfg, "pos_weight", None) == "auto":
            # valid pixels
            valid = pm.float().unsqueeze(1)  # [B,1,H,W]
            tgt = y_target                  # [B,1,H,W]
            n_pos = (tgt * valid).sum()
            n_valid = valid.sum()
            if n_pos > 0:
                w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, getattr(cfg, "pos_weight_clip", 20.0))
                pos_weight = torch.tensor([float(w.item())], device=device)
        elif isinstance(getattr(cfg, "pos_weight", None), (int, float)):
            pos_weight = torch.tensor([float(cfg.pos_weight)], device=device)

        # reconstruction loss (masked BCE or masked top-k BCE)
        if cfg.use_topk:
            recon = masked_topk_bce_with_logits(logits, y_target, pm, k_frac=cfg.k_frac, pos_weight=pos_weight)
        else:
            recon = masked_bce_with_logits(logits, y_target, pm, pos_weight=pos_weight)

        # reconstruction loss (masked BCE or masked top-k BCE)
        if cfg.use_topk:
            recon = masked_topk_bce_with_logits(logits, y_target, pm, k_frac=cfg.k_frac)
        else:
            recon = masked_bce_with_logits(logits, y_target, pm)

        # KL(q||p) averaged over batch
        kl = info["kl"].mean()

        # ELBO (beta=1 default)
        loss = recon + args.beta * kl

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # log scalars
        lr = optimizer.param_groups[0]["lr"]
        logger.log_scalars(step, {
            "loss": loss.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "lr": lr,
        })

        running["loss"] += float(loss.item())
        running["recon"] += float(recon.item())
        running["kl"] += float(kl.item())

        if step % 10 == 0:
            dt = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            n = 10.0
            print(
                f"[step {step:5d}] loss={running['loss']/n:.4f}  "
                f"recon={running['recon']/n:.4f}  kl={running['kl']/n:.4f}  "
                f"lr={lr:.6g}  bsz={cfg.batch_size}  chosen={info_sel.get('chosen_indices')[:4]}"
            )
            running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
            t0 = time.time()

    # param count sanity
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Finished sPU-Net smoke run. Params: {n_params}")

    # --- save checkpoint ---
    ckpt_path = args.outdir / args.save_name
    torch.save(
        {
            "step": int(args.max_steps),
            "model": model.state_dict(),
            "cfg": vars(cfg),
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to: {ckpt_path}")
    logger.close()

if __name__ == "__main__":
    main()
