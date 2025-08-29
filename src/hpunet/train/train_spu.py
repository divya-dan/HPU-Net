#!/usr/bin/env python3
"""
Standard PU-Net Training Script - reproducing the HPU-Net paper baseline

This trains the regular Probabilistic U-Net (sPU-Net) on LIDC data exactly
like the baseline described in the HPU-Net paper:

- Model: 5-scale U-Net, separate prior/posterior, global z in R^6, 3x1x1 combiner  
- Loss: ELBO with beta=1 -> loss = recon + KL (no fancy stuff)
- Reconstruction: sum over pixels, then mean over batch (mask out padded pixels)
- KL: sum over latent dims, then mean over batch
- Optimizer: Adam with weight decay, LR schedule from config
- Batch size: 32, Total steps: 240k (can override with --max-steps)

Note: We don't use GECO, stochastic top-k, beta-annealing, min-KL, or 
any other fancy stuff here. Just the clean baseline.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from dataclasses import is_dataclass, asdict
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from hpunet.utils.config import load_config
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.sched_optim import make_optimizer, make_scheduler
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.models.spu_net import sPUNet
from hpunet.utils.logger import Logger


def set_seed(seed: int):
    """set random seeds for reproducability"""
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def auto_pos_weight(y_target: Tensor, pad_mask: Tensor, clip: float = 20.0) -> Optional[Tensor]:
    """
    compute per-batch positive-class weight = negatives/positives
    returns scalar tensor on same device, or None if no positives
    """
    valid = pad_mask.float().unsqueeze(1)  # [B,1,H,W]
    n_pos = (y_target * valid).sum()
    n_valid = valid.sum()
    if float(n_pos) > 0:
        w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, clip)
        return torch.tensor([float(w.item())], device=y_target.device)
    return None


def recon_loss_sum_pixels_mean_batch(
    logits: Tensor, targets: Tensor, pad_mask: Tensor, pos_weight: Optional[Tensor] = None
) -> Tensor:
    """
    Binary cross-entropy with logits, masked properly
    - per-pixel BCE (no reduction)
    - mask out padded pixels
    - sum over H×W per sample  
    - mean over batch
    """
    assert logits.ndim == 4 and logits.size(1) == 1, "logits must be [B,1,H,W]"
    assert targets.shape == logits.shape, "targets must match logits"
    assert pad_mask.ndim == 3 and pad_mask.shape[0] == logits.shape[0]

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    mask = pad_mask.unsqueeze(1).to(bce.dtype)  # [B,1,H,W]
    bce = bce * mask
    
    # sum over pixels per sample, then mean over batch
    per_sample = bce.flatten(1).sum(dim=1)  # [B]
    return per_sample.mean()


def main():
    ap = argparse.ArgumentParser(description="Paper-faithful sPUNet training (LIDC)")
    ap.add_argument("--config", type=Path, required=True, help="Training config JSON file")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(), help="Project root directory")
    ap.add_argument("--data-root", type=Path, default=None, help="Data root directory (defaults to <project-root>/data/lidc_crops)")
    ap.add_argument("--max-steps", type=int, default=None, help="Override total training steps")
    ap.add_argument("--outdir", type=Path, default=Path("runs/spu_paper_exact"), help="Output directory")
    ap.add_argument("--save-name", type=str, default="spu_last.pth", help="Final checkpoint filename")
    args = ap.parse_args()

    # load config and setup
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.outdir.mkdir(parents=True, exist_ok=True)

    max_steps = int(args.max_steps if args.max_steps is not None else getattr(cfg, "total_steps", 240_000))
    eval_every_steps = int(getattr(cfg, "eval_every_steps", 5_000))
    ckpt_every_steps = int(getattr(cfg, "ckpt_every_steps", 10_000))

    data_root = args.data_root or (args.project_root / "data" / "lidc_crops")
    train_csv = data_root / "train.csv"

    # setup data
    train_ds = LIDCCropsDataset(
        csv_path=train_csv,
        project_root=args.project_root,
        image_size=128,
        augment=bool(getattr(cfg, "augment", True)),
        seed=int(getattr(cfg, "seed", 42)),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(getattr(cfg, "num_workers", 4)),
        pin_memory=True,
        drop_last=True,
    )

    # model setup - paper spec: in_ch=1, base=32, z_dim=6
    model = sPUNet(in_ch=1, base=32, z_dim=6).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer & LR schedule (paper uses weight decay everywhere)
    optimizer = make_optimizer(model.parameters(), cfg)
    scheduler = make_scheduler(optimizer, cfg)

    # target selection strategy: sample one grader per step
    recon_strategy = str(getattr(cfg, "recon_strategy", "random"))

    # pos weight handling (set cfg.pos_weight to "none" to disable)
    posw_mode = getattr(cfg, "pos_weight", "auto")
    posw_clip = float(getattr(cfg, "pos_weight_clip", 20.0))

    # for saving config into checkpoints
    if is_dataclass(cfg):
        cfg_to_save: Dict[str, Any] = asdict(cfg)
    elif isinstance(cfg, dict):
        cfg_to_save = dict(cfg)
    else:
        cfg_to_save = vars(cfg)

    # logger setup
    logger = Logger(args.outdir, use_tensorboard=True)

    print("Starting sPUNet training (paper baseline):")
    print(f"  Steps: {max_steps:,}")
    print(f"  Batch size: {cfg.batch_size}")
    print(f"  LR start: {cfg.lr} | WD: {cfg.weight_decay}")
    print(f"  Dataset size: {len(train_ds)}")

    model.train()
    data_iter = iter(train_loader)
    run_loss = run_recon = run_kl = 0.0

    for step in range(1, max_steps + 1):
        # get next batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        x = batch["image"].to(device, non_blocking=True)         # [B,1,H,W]
        y_all = batch["masks"].to(device, non_blocking=True)     # [B,4,H,W]
        pm = batch["pad_mask"].to(device, non_blocking=True)     # [B,H,W] (bool)

        # pick one target grader mask
        y_target, _sel = select_targets_from_graders(y_all, strategy=recon_strategy)

        # forward pass: use posterior during training
        logits, info = model(x, y_target=y_target, sample_posterior=True)

        # KL per-sample -> mean over batch
        kl = info.get("kl")
        if kl is None:
            kl_val = torch.zeros((), device=device)
        else:
            kl_val = kl.mean() if kl.ndim > 0 else kl

        # handle pos_weight
        if posw_mode == "auto":
            pos_weight = auto_pos_weight(y_target, pm, clip=posw_clip)
        elif isinstance(posw_mode, (int, float)):
            pos_weight = torch.tensor([float(posw_mode)], device=device)
        elif str(posw_mode).lower() == "none":
            pos_weight = None
        else:
            pos_weight = None

        # reconstruction: sum over pixels -> mean over batch, mask padding
        recon = recon_loss_sum_pixels_mean_batch(logits, y_target, pm, pos_weight)

        # paper ELBO (beta=1)
        loss = recon + kl_val

        # backprop
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        # track running losses
        run_loss += float(loss)
        run_recon += float(recon)
        run_kl += float(kl_val)

        logger.log_scalars(step, {
            "train/loss": float(loss),
            "train/recon": float(recon),
            "train/kl": float(kl_val),
            "train/lr": optimizer.param_groups[0]["lr"],
        })

        # print progress every 100 steps
        if step % 100 == 0:
            n = 100.0
            print(
                f"[step {step:5d}] loss={run_loss/n:.4f}  recon={run_recon/n:.4f}  kl={run_kl/n:.4f}  lr={optimizer.param_groups[0]['lr']:.6g}"
            )
            run_loss = run_recon = run_kl = 0.0

        # save checkpoint periodically
        if step % ckpt_every_steps == 0:
            ckpt_path = args.outdir / f"spu_step_{step}.pth"
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "cfg": cfg_to_save,
            }, ckpt_path)
            print(f"Checkpoint saved at step {step}: {ckpt_path}")

        # eval placeholder (could add validation here)
        if step % eval_every_steps == 0:
            pass

    # save final checkpoint
    ckpt_path = args.outdir / args.save_name
    torch.save({
        "step": int(max_steps),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "cfg": cfg_to_save,
    }, ckpt_path)
    print(f"Final checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()