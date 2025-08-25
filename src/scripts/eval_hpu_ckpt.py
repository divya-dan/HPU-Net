#!/usr/bin/env python3
from pathlib import Path
import argparse
import torch

# --- project imports ---
from hpunet.utils.logger import Logger
from scripts.hpu_end2end_debug import evaluate_hpu  # uses the function we just updated

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--num-images", type=int, default=64)
    ap.add_argument("--n-prior", type=int, default=16)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold")
    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    logger = Logger(args.outdir, use_tensorboard=False)

    evaluate_hpu(
        ckpt=args.ckpt,
        project_root=args.project_root,
        data_root=args.data_root,
        split=args.split,
        num_images=args.num_images,
        n_prior=args.n_prior,
        start_index=args.start_index,
        require_lesion=True,
        logger=logger,
        step=0,
        thr=args.thr,
    )

    # one-page summary image
    logger.save_summary_figure(args.outdir / "metrics_summary.png", ema=0.95)
    logger.close()

if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
