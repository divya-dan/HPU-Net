#!/usr/bin/env python3
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import gridspec as gspec

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.hpu_net import HPUNet
from hpunet.losses.metrics import binarize_logits


def _to_numpy_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x.clamp(0, 1).numpy()

def _show_bw(ax, mask: np.ndarray, title: str | None = None):
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if title: ax.set_title(title, fontsize=12)
    ax.axis("off")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n-prior", type=int, default=6)
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    ds = LIDCCropsDataset(
        csv_path=args.data_root / f"{args.split}.csv",
        project_root=args.project_root,
        image_size=128,
        augment=False
    )
    s   = ds[args.index]
    x   = s["image"].unsqueeze(0).to(device)   # [1,1,H,W]
    y   = s["masks"].to(device)                # [4,H,W]
    pad = s.get("pad_mask", torch.ones_like(y[0])).to(device)

    # model
    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    ckpt  = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    # ------- layout: single shared grid, right-aligned -------
    n_prior = int(args.n_prior)
    n_cols  = max(5, n_prior)      # shared #cols for ALL rows
    start5  = n_cols - 5           # start index of the [CT + 4 graders] block
    startP  = n_cols - n_prior     # start index of the prior row block

    fig = plt.figure(figsize=(22, 7.5), dpi=120)
    outer = gspec.GridSpec(3, 1, height_ratios=[1.0, 1.0, 1.0], hspace=0.35)

    # Row 0: CT + graders (right-aligned)
    g0 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[0], wspace=0.12)
    axes0 = [fig.add_subplot(g0[0, i]) for i in range(n_cols)]
    for ax in axes0: ax.axis("off")

    ct_np = _to_numpy_img(s["image"])
    axes0[start5].imshow(ct_np, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes0[start5].set_title("CT scan", fontsize=16)
    axes0[start5].axis("off")
    for g in range(4):
        gy = (y[g] > 0).float() * pad
        _show_bw(axes0[start5 + 1 + g], _to_numpy_img(gy), title=f"grader {g+1}")

    # Row title
    fig.text(0.02, 0.63, "i) Reconstructions", fontsize=18, weight="bold", va="center")

    # Row 1: Reconstructions under graders (right-aligned to same columns)
    g1 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[1], wspace=0.12)
    axes1 = [fig.add_subplot(g1[0, i]) for i in range(n_cols)]
    for ax in axes1: ax.axis("off")

    for g in range(4):
        y_t   = y[g:g+1].unsqueeze(0).float()            # [1,1,H,W]
        logits, _ = model(x, y_target=y_t, sample_posterior=True)
        pred  = binarize_logits(logits[0], thr=args.thr).to(torch.float32).squeeze() * pad
        _show_bw(axes1[start5 + 1 + g], _to_numpy_img(pred), title=f"{g+1}")

    # Row title
    fig.text(0.02, 0.305, "ii) Samples", fontsize=18, weight="bold", va="center")

    # Row 2: Prior samples (right-aligned)
    g2 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[2], wspace=0.12)
    axes2 = [fig.add_subplot(g2[0, i]) for i in range(n_cols)]
    for ax in axes2: ax.axis("off")

    for i in range(n_prior):
        logits, _ = model(x, y_target=None, sample_posterior=False)
        pred  = binarize_logits(logits[0], thr=args.thr).to(torch.float32).squeeze() * pad
        _show_bw(axes2[startP + i], _to_numpy_img(pred), title=f"{i+1}")

    fig.tight_layout(pad=0.5)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    plt.close(fig)

    # safe metadata print
    meta    = s.get("meta", {})
    patient = s.get("patient", meta.get("patient", "NA"))
    stem    = s.get("stem",    meta.get("stem",    "NA"))
    split   = s.get("split",   meta.get("split",   args.split))
    print(f"Saved panel to: {args.out}")
    print(f"Sample info — split: {split}, patient: {patient}, stem: {stem}")

if __name__ == "__main__":
    main()
