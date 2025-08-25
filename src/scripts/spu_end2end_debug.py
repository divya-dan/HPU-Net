#!/usr/bin/env python3
"""
End-to-end sPU-Net debug driver:
- train for N steps
- evaluate on a subset
- render a black&white grid figure (CT | graders | reconstructions | samples)

Usage (example):
  export PYTHONPATH="$(pwd)/src"
  python src/scripts/spu_end2end_debug.py \
      --project-root "$(pwd)" \
      --data-root "data/lidc_crops" \
      --max-steps 2000 \
      --batch-size 16 \
      --n-cols 8 \
      --n-sample-rows 3 \
      --outdir "runs/spu_debug"

The figure will be saved to: <outdir>/fig_grid.png
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ensure src on path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import ProbUNet
from hpunet.losses.topk_ce import masked_bce_with_logits
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.metrics import binarize_logits, iou_binary, hungarian_matched_iou, ged2
from hpunet.utils.logger import Logger


def set_seed(seed: int = 123):
    import numpy as _np
    random.seed(seed); _np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def auto_pos_weight(y_target: torch.Tensor, pad_mask: torch.Tensor, clip: float = 20.0) -> torch.Tensor | None:
    """
    Per-batch positive class weight for BCE: (negatives / positives).
    y_target: [B,1,H,W] (0/1), pad_mask: [B,H,W] (bool)
    """
    valid = pad_mask.float().unsqueeze(1)  # [B,1,H,W]
    n_pos = (y_target * valid).sum()
    n_valid = valid.sum()
    if n_pos > 0:
        w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, clip)
        return torch.tensor([float(w.item())], device=y_target.device)
    return None


@torch.no_grad()
def pick_lesion_indices(ds: LIDCCropsDataset, n: int, start: int = 0) -> list[int]:
    out = []
    i = start
    while len(out) < n and i < len(ds):
        m = ds[i]["masks"] > 0
        if m.sum().item() > 0:
            out.append(i)
        i += 1
    # if not enough lesion tiles, just fill with the last ones
    while len(out) < n and len(out) > 0:
        out.append(out[-1])
    return out


def train_spu_for_steps(
    project_root: Path, data_root: Path, outdir: Path,
    max_steps: int = 2000, batch_size: int = 16, lr: float = 1e-4,
    seed: int = 123, num_workers: int = 2, logger: Logger | None = None,
    ) -> tuple[Path, Logger]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logger or Logger(outdir, use_tensorboard=True)

    logger.log_scalars(0, {
        "train/loss": 0.0, "train/recon": 0.0, "train/kl": 0.0, "train/lr": lr,
        "eval/num": 0.0, "eval/IoU_rec": 0.0, "eval/HungIoU": 0.0, "eval/GED2": 0.0,
    })

    # data
    train_csv = data_root / "train.csv"
    train_ds = LIDCCropsDataset(
        csv_path=train_csv, project_root=project_root,
        image_size=128, augment=True, seed=seed
    )
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)

    # model/opt
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    it = iter(loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x = batch["image"].to(device, non_blocking=True)        # [B,1,H,W]
        y_all = batch["masks"].to(device, non_blocking=True)    # [B,4,H,W]
        pm = batch["pad_mask"].to(device, non_blocking=True)    # [B,H,W]

        # choose a grader target
        y_target, _ = select_targets_from_graders(y_all, strategy="random")  # [B,1,H,W]

        # forward using posterior
        logits, info = model(x, y_target=y_target, sample_posterior=True)

        # KL: ensure scalar
        kl = info["kl"]
        if isinstance(kl, torch.Tensor) and kl.ndim > 0:
            kl = kl.mean()

        # class balancing
        pos_w = auto_pos_weight(y_target, pm, clip=20.0)

        # reconstruction (scalar)
        recon = masked_bce_with_logits(logits, y_target, pm, pos_weight=pos_w)

        # total loss MUST be scalar
        loss = (recon + kl).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # log every step (cheap), and also print periodically
        logger.log_scalars(step, {
            "train/loss": float(loss),
            "train/recon": float(recon),
            "train/kl": float(kl),
            "train/lr": opt.param_groups[0]["lr"],
        })
        if step % 50 == 0 or step == 1:
            print(f"[step {step:5d}] loss={float(loss):.4f}  recon={float(recon):.4f}  kl={float(kl):.4f}")

    # save checkpoint
    ckpt = outdir / "spu_last.pth"
    torch.save({"step": max_steps, "model": model.state_dict()}, ckpt)
    print(f"Checkpoint saved to: {ckpt}")
    return ckpt, logger


@torch.no_grad()
def evaluate_spu(ckpt: Path, project_root: Path, data_root: Path,
                 split: str = "test", num_images: int = 64, n_prior: int = 16,
                 start_index: int = 0, require_lesion: bool = True,
                 logger: Logger | None = None, step: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LIDCCropsDataset(csv_path=data_root / f"{split}.csv",
                          project_root=project_root, image_size=128, augment=False)
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    vals_rec, vals_hung, vals_ged = [], [], []
    kept, idx = 0, start_index
    while kept < num_images and idx < len(ds):
        sample = ds[idx]
        x = sample["image"].unsqueeze(0).to(device)
        y = sample["masks"].to(device)              # [4,H,W]
        gt_set = (y > 0).to(torch.uint8)
        if require_lesion and gt_set.sum().item() == 0:
            idx += 1
            continue

        # posterior recon IoU per grader
        per_g = []
        for g in range(4):
            y_t = y[g:g+1].unsqueeze(0).float()
            logits, _ = model(x, y_target=y_t, sample_posterior=True)
            pred = binarize_logits(logits[0])
            per_g.append(iou_binary(pred, gt_set[g:g+1]).item())
        vals_rec.append(sum(per_g) / 4.0)

        # prior samples
        preds = []
        for _ in range(n_prior):
            logits, _ = model(x, y_target=None, sample_posterior=False)
            preds.append(binarize_logits(logits[0]).squeeze(0))
        pred_set = torch.stack(preds, dim=0)
        vals_hung.append(hungarian_matched_iou(gt_set, pred_set))
        vals_ged.append(max(ged2(gt_set, pred_set), 0.0))  # clip ≥ 0

        kept += 1
        idx += 1

    def _m(v): return sum(v) / max(len(v), 1)
    iou_rec = _m(vals_rec); hung = _m(vals_hung); ged = _m(vals_ged)
    print(f"[{split}] N={kept} | IoU_rec={iou_rec:.4f} | Hung.IoU={hung:.4f} | GED^2={ged:.4f}")
    if logger is not None:
        s = 0 if step is None else int(step)
        logger.log_scalars(s, {
            "eval/num": float(kept),
            "eval/IoU_rec": float(iou_rec),
            "eval/HungIoU": float(hung),
            "eval/GED2": float(ged),
        })


@torch.no_grad()
def render_grid(
    ckpt: Path, project_root: Path, data_root: Path, out_png: Path,
    split: str = "test", n_cols: int = 8, n_sample_rows: int = 3, start_index: int = 0,
    dividers: bool = True, divider_color: str = "red", divider_lw: float = 3.0,
):
    """
    Create a grid with:
      row 0: CT scans
      rows 1..4: 4 expert graders
      rows 5..8: 4 posterior reconstructions (targeting each grader)
      rows 9..(9+n_sample_rows-1): prior samples
    All masks are white on black; CT is grayscale.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LIDCCropsDataset(csv_path=data_root / f"{split}.csv",
                          project_root=project_root, image_size=128, augment=False)
    idxs = pick_lesion_indices(ds, n_cols, start=start_index)
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    # prepare per-column data
    imgs = []            # [n_cols, H, W] float in [0,1]
    graders = []         # [n_cols, 4, H, W] bool
    recons = []          # [n_cols, 4, H, W] bool
    samples = []         # [n_sample_rows, n_cols, H, W] bool

    for j, idx in enumerate(idxs):
        s = ds[idx]
        x = s["image"].unsqueeze(0).to(device)                # [1,1,H,W]
        y = s["masks"] > 0                                    # [4,H,W] bool
        pm = s["pad_mask"].bool()                             # [H,W]
        imgs.append(s["image"][0].numpy())                    # [H,W] (0..1)
        graders.append((y & pm).numpy())                      # apply pad

        # posterior recon per grader
        per_g = []
        for g in range(4):
            y_t = y[g:g+1].float().unsqueeze(0).to(device)    # [1,1,H,W]
            logits, _ = model(x, y_target=y_t, sample_posterior=True)
            pred = (logits.sigmoid()[0, 0] > 0.5).detach().cpu().numpy()
            per_g.append(pred & pm.numpy())
        recons.append(np.stack(per_g, 0))

        # prior samples for this image (one per sample row)
        col_samples = []
        for _ in range(n_sample_rows):
            l, _ = model(x, y_target=None, sample_posterior=False)
            pr = (l.sigmoid()[0, 0] > 0.5).detach().cpu().numpy()
            col_samples.append(pr & pm.numpy())
        samples.append(np.stack(col_samples, 0))  # [n_sample_rows, H, W]

    imgs = np.stack(imgs, 0)                       # [C,H,W]
    graders = np.stack(graders, 0)                 # [C,4,H,W]
    recons = np.stack(recons, 0)                   # [C,4,H,W]
    samples = np.stack(samples, 0)                 # currently [C,R,H,W]
    samples = np.transpose(samples, (1, 0, 2, 3))  # -> [R,C,H,W]

    H, W = imgs.shape[-2:]
    n_rows = 1 + 4 + 4 + n_sample_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7*n_cols, 1.7*n_rows))
    # make sure axes is 2D array even if n_cols==1
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    # Row 0: CT
    for c in range(n_cols):
        ax = axes[0, c]; ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 1..4: graders
    for r in range(4):
        for c in range(n_cols):
            ax = axes[1+r, c]
            ax.imshow(graders[c, r], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 5..8: posterior reconstructions
    for r in range(4):
        for c in range(n_cols):
            ax = axes[5+r, c]
            ax.imshow(recons[c, r], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 9..: prior samples
    for r in range(n_sample_rows):
        for c in range(n_cols):
            ax = axes[9+r, c]
            ax.imshow(samples[r, c], cmap="gray", vmin=0, vmax=1); ax.axis("off")

    # Section labels on the left
    fig.text(0.005, 1.00 - (0.5 / n_rows), "CT scans", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((1 + 2.0) / n_rows), "4 expert graders", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((5 + 2.0) / n_rows), "reconstructions", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((9 + (n_sample_rows/2)) / n_rows), "samples", va="center", ha="left", rotation=90, fontsize=10)

    fig.tight_layout(rect=[0.03, 0.02, 1, 0.98])
    # --- section dividers (between CT | graders | reconstructions) ---
    if dividers:
        # ensure final axes positions are computed
        plt.draw()
        # draw a horizontal line between row r and r+1 in figure coords
        def _hline_between(r: int):
            if r + 1 >= n_rows: 
                return
            # left/right bounds spanning the grid
            left = min(axes[i, 0].get_position().x0 for i in range(n_rows))
            right = max(axes[i, -1].get_position().x1 for i in range(n_rows))
            y_bot_this = axes[r, 0].get_position().y0
            y_top_next = axes[r + 1, 0].get_position().y1
            y = 0.5 * (y_bot_this + y_top_next)
            fig.add_artist(Line2D([left, right], [y, y],
                                  transform=fig.transFigure,
                                  color=divider_color, linewidth=divider_lw))
        # after CT row (0), after graders (row 4), after recons (row 8)
        for r in (0, 4, 8):
            if r < n_rows - 1:
                _hline_between(r)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid figure to: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("runs/spu_debug"))
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--n-cols", type=int, default=8, help="number of images (columns) in the grid")
    ap.add_argument("--n-sample-rows", type=int, default=3, help="rows of prior samples per image")
    ap.add_argument("--start-index", type=int, default=0, help="start scanning the split from this index")
    ap.add_argument("--eval-num", type=int, default=64)
    args = ap.parse_args()

    project_root = args.project_root
    data_root = args.data_root or (project_root / "data" / "lidc_crops")
    outdir = args.outdir

    ckpt, logger = train_spu_for_steps(
        project_root, data_root, outdir,
        max_steps=args.max_steps, batch_size=args.batch_size,
        lr=1e-4, seed=123, num_workers=args.num_workers,
    )

    # quick eval on lesion-only subset
    evaluate_spu(
        ckpt, project_root, data_root,
        split="test", num_images=args.eval_num, n_prior=16,
        start_index=args.start_index, require_lesion=True,
        logger=logger, step=args.max_steps
    )

    # figure like the paper (black background + white masks)
    render_grid(
        ckpt, project_root, data_root,
        out_png=outdir / "fig_grid.png",
        split="test", n_cols=args.n_cols,
        n_sample_rows=args.n_sample_rows, start_index=args.start_index
    )
    logger.save_summary_figure(outdir / "metrics_summary.png", ema=0.95)
    logger.close()


if __name__ == "__main__":
    
    main()
