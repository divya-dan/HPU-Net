from __future__ import annotations
import argparse, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from hpunet.utils.config import load_config
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.sched_optim import make_optimizer, make_scheduler
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.topk_ce import masked_bce_with_logits, masked_topk_bce_with_logits
from hpunet.losses.geco import GECO, GECOConfig
from hpunet.models.hpu_net import HPUNet
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
    ap.add_argument("--outdir", type=Path, default=Path("runs/hpu"))
    ap.add_argument("--save-name", type=str, default="hpu_last.pth")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root or (args.project_root / "data" / "lidc_crops")
    train_csv = data_root / "train.csv"

    # Data
    train_ds = LIDCCropsDataset(
        csv_path=train_csv, project_root=args.project_root,
        image_size=128, augment=cfg.augment, seed=cfg.seed
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    # Model / GECO / opt / sched
    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    gcfg = cfg.geco or {"kappa": 0.05, "alpha": 0.99, "lambda_init": 1.0}
    geco = GECO(GECOConfig(**gcfg)).to(device)
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    running = {"L": 0.0, "recon": 0.0, "kl": 0.0, "lam": 0.0, "C": 0.0, "Cbar": 0.0}
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

        # choose target grader
        y_target, _ = select_targets_from_graders(y_all, strategy=cfg.recon_strategy)

        # forward with posterior sampling (training)
        logits, info = model(x, y_target=y_target, sample_posterior=True)

        # reconstruction (top-k by default for HPU)
        if cfg.use_topk:
            recon = masked_topk_bce_with_logits(logits, y_target, pm, k_frac=cfg.k_frac)
        else:
            recon = masked_bce_with_logits(logits, y_target, pm)

        kl = info["kl"]  # scalar

        # GECO update (uses recon only)
        ge = geco.step(recon.detach())
        L = geco.lagrangian(recon, kl)

        optimizer.zero_grad(set_to_none=True)
        L.backward()
        optimizer.step()
        scheduler.step()

        # log scalars
        lr = optimizer.param_groups[0]["lr"]
        logger.log_scalars(step, {
            "L": L.item(),
            "recon": recon.item(),
            "kl": kl.item(),
            "lambda": ge["lambda"],
            "C": ge["C"],
            "C_bar": ge["C_bar"],
            "lr": lr,
        })

        running["L"] += float(L.item())
        running["recon"] += float(recon.item())
        running["kl"] += float(kl.item())
        running["lam"] += ge["lambda"]
        running["C"] += ge["C"]
        running["Cbar"] += ge["C_bar"]

        if step % 10 == 0:
            n = 10.0
            lr = optimizer.param_groups[0]["lr"]
            print(
                f"[step {step:5d}] L={running['L']/n:.4f} recon={running['recon']/n:.4f} "
                f"kl={running['kl']/n:.4f} lam={running['lam']/n:.3f} "
                f"C={running['C']/n:.4f} C̄={running['Cbar']/n:.4f} lr={lr:.6g}"
            )
            running = {"L": 0.0, "recon": 0.0, "kl": 0.0, "lam": 0.0, "C": 0.0, "Cbar": 0.0}
            t0 = time.time()

    # Save checkpoint
    ckpt_path = args.outdir / args.save_name
    torch.save({"step": int(args.max_steps), "model": model.state_dict(), "cfg": vars(cfg)}, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")
    logger.close()

if __name__ == "__main__":
    main()
