#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import gridspec as gspec

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.hpu_net import HPUNet
from hpunet.models.spu_net import ProbUNet
from hpunet.losses.metrics import iou_binary, hungarian_matched_iou


# --------------------- utils ---------------------

@torch.no_grad()
def _to_np(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().float()
    while x.ndim > 2 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x.clamp(0, 1).numpy()

def _binarize_from_logits(logits: torch.Tensor, thr: float) -> torch.Tensor:
    # logits [B,1,H,W] -> prob -> bin -> [B,1,H,W] in {0,1}
    return (torch.sigmoid(logits) >= thr).to(logits.dtype)

def _gt_set_from_y(y: torch.Tensor) -> torch.Tensor:
    # y [4,H,W] -> [4,H,W] uint8 {0,1}
    return (y > 0).to(torch.uint8)

def _require_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_model(model_type: str, ckpt: Path, device) -> torch.nn.Module:
    if model_type == "hpu":
        model = HPUNet(in_ch=1, base=24, z_ch=8).to(device)
    elif model_type == "spu":
        model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    else:
        raise ValueError("model_type must be 'hpu' or 'spu'")
    obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(obj["model"], strict=True)
    model.eval()
    return model

# --------------------- threshold search ---------------------

@torch.no_grad()
def eval_best_threshold(
    model: torch.nn.Module,
    dataset: LIDCCropsDataset,
    device,
    metric: str = "recon",       # 'recon' or 'hungarian'
    n_prior: int = 16,
    require_lesion: bool = True,
    num_images: int = 64,
    thr_grid: tuple[float,float,float] = (0.05, 0.95, 0.05),
    model_type: str = "hpu",
) -> tuple[float, dict]:
    """
    Returns (best_thr, summary) where summary contains per-threshold scores.
    - recon: mean posterior-reconstruction IoU (average over 4 graders, then over images)
    - hungarian: mean Hungarian-matched IoU between prior-sample set and graders
    """
    thrs = np.arange(*thr_grid)
    scores = np.zeros_like(thrs, dtype=np.float64)

    kept, idx = 0, 0
    while kept < num_images and idx < len(dataset):
        s = dataset[idx]
        idx += 1

        x = s["image"].unsqueeze(0).to(device)   # [1,1,H,W]
        y = s["masks"].to(device)                # [4,H,W]
        pad = s.get("pad_mask", torch.ones_like(y[0], dtype=torch.uint8)).to(device)
        gt_set = _gt_set_from_y(y) * pad        # [4,H,W]

        if require_lesion and gt_set.sum().item() == 0:
            continue

        if metric == "recon":
            # For each grader, posterior recon, IoU vs that grader
            # We binarize with each threshold and average over graders
            logits_per_g = []
            for g in range(4):
                y_t = y[g:g+1].unsqueeze(0).float()
                logits, _ = model(x, y_target=y_t, sample_posterior=True)
                logits_per_g.append(logits.detach().clone())  # [1,1,H,W]
            for j, t in enumerate(thrs):
                ious = []
                for g in range(4):
                    pred = _binarize_from_logits(logits_per_g[g], float(t))[0,0] * pad
                    ious.append(iou_binary(pred, gt_set[g:g+1]).item())
                scores[j] += float(np.mean(ious))

        elif metric == "hungarian":
            # Build set of prior samples (n_prior), then Hungarian IoU vs grader set
            logits_list = []
            for _ in range(n_prior):
                logits, _ = model(x, y_target=None, sample_posterior=False)
                logits_list.append(logits.detach().clone())
            # convert once per threshold
            for j, t in enumerate(thrs):
                preds = []
                for lg in logits_list:
                    preds.append((_binarize_from_logits(lg, float(t))[0,0] * pad).to(torch.uint8))
                pred_set = torch.stack(preds, dim=0)  # [K,H,W]
                scores[j] += float(hungarian_matched_iou(gt_set, pred_set))
        else:
            raise ValueError("metric must be 'recon' or 'hungarian'")

        kept += 1

    if kept == 0:
        # fallback
        return 0.5, {"thrs": thrs.tolist(), "scores": scores.tolist(), "N": 0}

    scores /= kept
    best_idx = int(np.argmax(scores))
    best_thr = float(thrs[best_idx])
    return best_thr, {"thrs": thrs.tolist(), "scores": scores.tolist(), "best": best_thr, "N": int(kept)}


# --------------------- figure renderers ---------------------

def _show_bw(ax, arr: np.ndarray, title: str | None = None, fontsize: int = 12):
    ax.imshow(arr, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    if title: ax.set_title(title, fontsize=fontsize)
    ax.axis("off")

@torch.no_grad()
def render_debug_grid(
    model, sample, device, out_path: Path,
    n_cols: int = 8, n_sample_rows: int = 3, n_prior: int = 16, thr: float = 0.5
):
    """
    Debug grid:
      Row 0: CT + 4 graders
      Row 1: Posterior recons for graders 1..4 (with threshold)
      Rows 2..: prior samples laid out in rows
    """
    x = sample["image"].unsqueeze(0).to(device)
    y = sample["masks"].to(device)
    pad = sample.get("pad_mask", torch.ones_like(y[0], dtype=torch.uint8)).to(device)

    fig = plt.figure(figsize=(n_cols*2.2, (2+n_sample_rows)*2.2), dpi=120)
    outer = gspec.GridSpec(2 + n_sample_rows, n_cols, hspace=0.25, wspace=0.1)

    # Row 0
    ax = fig.add_subplot(outer[0, 0]); _show_bw(ax, _to_np(sample["image"]), "CT scan", fontsize=14)
    for g in range(4):
        ax = fig.add_subplot(outer[0, 1+g])
        _show_bw(ax, _to_np(((y[g]>0).float()*pad)), f"grader {g+1}", fontsize=12)

    # Posterior recons (first 4 cells of row 1)
    for g in range(4):
        y_t = y[g:g+1].unsqueeze(0).float()
        logits, _ = model(x, y_target=y_t, sample_posterior=True)
        pred = (_binarize_from_logits(logits, thr)[0,0] * pad).float()
        ax = fig.add_subplot(outer[1, g])
        _show_bw(ax, _to_np(pred), f"recon {g+1}", fontsize=12)

    # Prior samples grid
    K = n_cols * n_sample_rows
    preds = []
    for _ in range(K):
        logits, _ = model(x, y_target=None, sample_posterior=False)
        preds.append((_binarize_from_logits(logits, thr)[0,0] * pad).float())
    preds = torch.stack(preds, dim=0)  # [K,H,W]
    k = 0
    for r in range(n_sample_rows):
        for c in range(n_cols):
            ax = fig.add_subplot(outer[2+r, c])
            _show_bw(ax, _to_np(preds[k]), f"s{k+1}", fontsize=10)
            k += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)

@torch.no_grad()
def render_fig2_style(
    model, sample, device, out_path: Path, n_prior: int = 6, thr: float = 0.5
):
    """
    Figure-2 style, right-aligned B/W:
      Row i)   CT + graders (right-aligned 5-block)
      Row ii)  Recon 1..4 under grader 1..4 (right-aligned)
      Row iii) n_prior samples, right-aligned
    """
    x = sample["image"].unsqueeze(0).to(device)
    y = sample["masks"].to(device)
    pad = sample.get("pad_mask", torch.ones_like(y[0], dtype=torch.uint8)).to(device)

    n_cols = max(5, n_prior)
    start5 = n_cols - 5
    startP = n_cols - n_prior

    fig = plt.figure(figsize=(22, 7.5), dpi=120)
    outer = gspec.GridSpec(3, 1, height_ratios=[1,1,1], hspace=0.35)

    # Row 0
    g0 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[0], wspace=0.12)
    axes0 = [fig.add_subplot(g0[0, i]) for i in range(n_cols)]
    for ax in axes0: ax.axis("off")
    _show_bw(axes0[start5], _to_np(sample["image"]), "CT scan", fontsize=16)
    for g in range(4):
        _show_bw(axes0[start5 + 1 + g], _to_np(((y[g]>0).float()*pad)), f"grader {g+1}", fontsize=12)
    fig.text(0.02, 0.63, "i) Reconstructions", fontsize=18, weight="bold", va="center")

    # Row 1 (recons)
    g1 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[1], wspace=0.12)
    axes1 = [fig.add_subplot(g1[0, i]) for i in range(n_cols)]
    for ax in axes1: ax.axis("off")
    for g in range(4):
        y_t = y[g:g+1].unsqueeze(0).float()
        logits, _ = model(x, y_target=y_t, sample_posterior=True)
        pred = (_binarize_from_logits(logits, thr)[0,0] * pad).float()
        _show_bw(axes1[start5 + 1 + g], _to_np(pred), f"{g+1}", fontsize=12)
    fig.text(0.02, 0.305, "ii) Samples", fontsize=18, weight="bold", va="center")

    # Row 2 (prior samples)
    g2 = gspec.GridSpecFromSubplotSpec(1, n_cols, subplot_spec=outer[2], wspace=0.12)
    axes2 = [fig.add_subplot(g2[0, i]) for i in range(n_cols)]
    for ax in axes2: ax.axis("off")
    for i in range(n_prior):
        logits, _ = model(x, y_target=None, sample_posterior=False)
        pred = (_binarize_from_logits(logits, thr)[0,0] * pad).float()
        _show_bw(axes2[startP + i], _to_np(pred), f"{i+1}", fontsize=12)

    fig.tight_layout(pad=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig)


# --------------------- main ---------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["hpu", "spu"], required=True)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)

    # threshold search
    ap.add_argument("--thr-split", type=str, default="val")
    ap.add_argument("--thr-metric", choices=["recon", "hungarian"], default="recon")
    ap.add_argument("--thr-num-images", type=int, default=64)
    ap.add_argument("--n-prior", type=int, default=16)
    ap.add_argument("--require-lesion", action="store_true", default=False)
    ap.add_argument("--thr-grid", type=str, default="0.05,0.95,0.05")  # start,stop,step

    # panel generation
    ap.add_argument("--panel-split", type=str, default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--debug-cols", type=int, default=8)
    ap.add_argument("--debug-rows", type=int, default=3)
    ap.add_argument("--fig2-samples", type=int, default=6)

    ap.add_argument("--outdir", type=Path, required=True)
    args = ap.parse_args()

    device = _require_device()

    # datasets
    thr_ds = LIDCCropsDataset(
        csv_path=args.data_root / f"{args.thr_split}.csv",
        project_root=args.project_root,
        image_size=128, augment=False
    )
    panel_ds = LIDCCropsDataset(
        csv_path=args.data_root / f"{args.panel_split}.csv",
        project_root=args.project_root,
        image_size=128, augment=False
    )

    # model
    model = _load_model(args.model, args.ckpt, device)

    # threshold search
    a,b,c = [float(x) for x in args.thr_grid.split(",")]
    best_thr, summary = eval_best_threshold(
        model, thr_ds, device,
        metric=args.thr_metric,
        n_prior=args.n_prior,
        require_lesion=args.require_lesion,
        num_images=args.thr_num_images,
        thr_grid=(a,b,c),
        model_type=args.model,
    )
    print(f"Best threshold ({args.thr_metric}, split={args.thr_split}, N={summary['N']}): {best_thr:.2f}")

    # sample for panels
    s = panel_ds[int(args.index)]

    # outputs
    args.outdir.mkdir(parents=True, exist_ok=True)
    out_debug = args.outdir / f"{args.model}_debug_{args.panel_split}_idx{args.index}.png"
    out_fig2  = args.outdir / f"{args.model}_fig2_{args.panel_split}_idx{args.index}.png"

    # render both
    render_debug_grid(
        model, s, device, out_debug,
        n_cols=args.debug_cols, n_sample_rows=args.debug_rows,
        n_prior=args.n_prior, thr=best_thr
    )
    render_fig2_style(
        model, s, device, out_fig2,
        n_prior=args.fig2_samples, thr=best_thr
    )

    # print meta
    meta    = s.get("meta", {})
    patient = s.get("patient", meta.get("patient", "NA"))
    stem    = s.get("stem",    meta.get("stem",    "NA"))
    split   = s.get("split",   meta.get("split",   args.panel_split))
    print(f"Saved: {out_debug.name} and {out_fig2.name}")
    print(f"Sample info — split: {split}, patient: {patient}, stem: {stem}")


if __name__ == "__main__":
    main()
