#!/usr/bin/env python3
"""
sPUNet Training Script - Corrected for LIDC Paper Specification
Standard Probabilistic U-Net with ELBO loss and separate prior/posterior networks.
"""

from __future__ import annotations
import argparse, time
from pathlib import Path
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from hpunet.utils.config import load_config
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.sched_optim import make_optimizer, make_scheduler
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.topk_ce import masked_bce_with_logits
from hpunet.models.spu_net import sPUNet  # FIXED: Import corrected model
from hpunet.utils.logger import Logger
from dataclasses import is_dataclass, asdict


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def auto_pos_weight(y_target: torch.Tensor, pad_mask: torch.Tensor, clip: float = 20.0) -> torch.Tensor | None:
    """Per-batch positive-class weight for BCE (negatives / positives)."""
    valid = pad_mask.float().unsqueeze(1)  # [B,1,H,W]
    n_pos = (y_target * valid).sum()
    n_valid = valid.sum()
    if n_pos > 0:
        w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, clip)
        return torch.tensor([float(w.item())], device=y_target.device)
    return None


def main():
    ap = argparse.ArgumentParser(description="sPUNet Training for LIDC Dataset")
    ap.add_argument("--config", type=Path, required=True, help="Training config JSON file")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    ap.add_argument("--data-root", type=Path, default=None, help="Data root directory")
    ap.add_argument("--max-steps", type=int, default=None, help="Max training steps (overrides config)")
    ap.add_argument("--beta", type=float, default=None, help="ELBO beta for KL term (overrides config)")
    ap.add_argument("--outdir", type=Path, default=Path("runs/spu"), help="Output directory")
    ap.add_argument("--save-name", type=str, default="spu_last.pth", help="Final checkpoint name")
    args = ap.parse_args()

    # Load config and setup
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    # FIXED: Use config total_steps if max_steps not specified
    max_steps = args.max_steps if args.max_steps is not None else getattr(cfg, 'total_steps', 240000)
    eval_every_steps = getattr(cfg, 'eval_every_steps', 5000)
    ckpt_every_steps = getattr(cfg, 'ckpt_every_steps', 10000)
    
    # ELBO beta parameter
    beta = args.beta if args.beta is not None else getattr(cfg, 'beta', 1.0)

    data_root = args.data_root or (args.project_root / "data" / "lidc_crops")
    train_csv = data_root / "train.csv"

    # Data loading
    train_ds = LIDCCropsDataset(
        csv_path=train_csv, 
        project_root=args.project_root,
        image_size=128, 
        augment=cfg.augment, 
        seed=cfg.seed
    )
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=cfg.num_workers, 
        pin_memory=True, 
        drop_last=True
    )

    # FIXED: Model with correct parameters for sPUNet
    model = sPUNet(in_ch=1, base=32, z_dim=6).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer and scheduler
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    # Training configuration
    recon_strategy = getattr(cfg, "recon_strategy", "random")
    posw_mode = getattr(cfg, "pos_weight", "auto")
    posw_clip = float(getattr(cfg, "pos_weight_clip", 20.0))

    # Prepare config for saving (used in periodic checkpoints)
    if is_dataclass(cfg):
        cfg_to_save = asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_to_save = cfg
    else:
        cfg_to_save = vars(cfg)

    # Training state
    running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
    logger = Logger(args.outdir, use_tensorboard=True)

    print(f"Starting sPUNet training:")
    print(f"  Max steps: {max_steps:,}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  Learning rate: {cfg.lr}")
    print(f"  ELBO beta: {beta}")
    print(f"  Dataset: {len(train_ds)} training examples")

    model.train()
    data_iter = iter(train_loader)
    
    for step in range(1, int(max_steps) + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        # Load batch data
        x = batch["image"].to(device, non_blocking=True)        # [B,1,H,W]
        y_all = batch["masks"].to(device, non_blocking=True)    # [B,4,H,W]
        pm = batch["pad_mask"].to(device, non_blocking=True)    # [B,H,W]

        # Select target grader mask (randomly among 4 graders)
        y_target, info_sel = select_targets_from_graders(y_all, strategy=recon_strategy)

        # Forward pass: sample from posterior q(z|x,y) during training
        logits, info = model(x, y_target=y_target, sample_posterior=True)

        # FIXED: Extract KL divergence (use consistent key)
        kl = info.get("KL_sum", torch.tensor(0.0, device=device))
        if isinstance(kl, torch.Tensor) and kl.ndim > 0:
            kl = kl.mean()

        # Compute positive weight for class imbalance
        pos_weight = None
        if posw_mode == "auto":
            pos_weight = auto_pos_weight(y_target, pm, clip=posw_clip)
        elif isinstance(posw_mode, (int, float)):
            pos_weight = torch.tensor([float(posw_mode)], device=device)

        # Reconstruction loss: mean BCE over valid pixels
        recon = masked_bce_with_logits(logits, y_target, pm, pos_weight=pos_weight)

        # ELBO loss: L = E[recon] + β * KL(q||p) 
        loss = recon + beta * kl

        # Optimization step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Logging
        lr = optimizer.param_groups[0]["lr"]
        logger.log_scalars(step, {
            "train/loss": float(loss),
            "train/recon": float(recon),
            "train/kl": float(kl),
            "train/lr": lr,
            "train/beta": beta,
        })

        # Running averages for console output
        running["loss"] += float(loss)
        running["recon"] += float(recon)
        running["kl"] += float(kl)

        # Console logging
        if step % 100 == 0:
            n = 100.0
            print(
                f"[step {step:5d}] loss={running['loss']/n:.4f}  "
                f"recon={running['recon']/n:.4f}  kl={running['kl']/n:.4f}  "
                f"lr={lr:.6g}  chosen={info_sel.get('chosen_indices', ['?'])[:4]}"
            )
            running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}

        # ADDED: Periodic checkpointing
        if step % ckpt_every_steps == 0:
            ckpt_path = args.outdir / f"spu_step_{step}.pth"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg_to_save
            }, ckpt_path)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")

        # ADDED: Periodic evaluation (placeholder - implement validation if you have val data)
        if step % eval_every_steps == 0:
            print(f"[step {step:5d}] Evaluation placeholder (add validation logic here)")
            # TODO: Add validation loop here if you have validation data

    # Save final checkpoint
    ckpt_path = args.outdir / args.save_name
    torch.save({
        "step": int(max_steps),
        "model": model.state_dict(), 
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg_to_save
    }, ckpt_path)
    print(f"Final checkpoint saved to: {ckpt_path}")
    logger.close()


if __name__ == "__main__":
    main()