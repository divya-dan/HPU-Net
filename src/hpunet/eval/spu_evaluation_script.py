#!/usr/bin/env python3
# sPUNet Evaluation Script (Updated to match HPU script v3)
# - Shows raw CT (no masking on input image).
# - Computes IoU (empty-vs-empty => 1) and IoU-based GED^2 for reconstructions vs graders.
# - Reports metrics for (A) all cases in the CSV scope and (B) only cases with all 4 graders.
# - Two visualization styles; grader and reconstruction columns are aligned.
# - All titles and footers right-aligned; IoU printed under each reconstruction.
# - Switches: --csv-name (train.csv/test.csv), --eval-scope (all/visualize_only), --ged-clamp-nonneg.
# - Saves metrics CSVs.

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Dict

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import csv

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import sPUNet
from hpunet.utils.config import load_config


# ----------------------------
# Model loading
# ----------------------------

def load_model(ckpt_path: Path, device: torch.device) -> sPUNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    # NOTE: adjust defaults if your sPUNet signature differs
    model = sPUNet(in_ch=1, base=32, z_dim=6).to(device)
    model.load_state_dict(ckpt["model"])  # expects key 'model'
    model.eval()
    print(f"Loaded sPUNet from step {ckpt.get('step', 'unknown')}")
    return model


# ----------------------------
# Sampling helpers
# ----------------------------

@torch.no_grad()
def generate_posterior_reconstructions(
    model: sPUNet,
    image: torch.Tensor,
    grader_masks: torch.Tensor,
) -> List[torch.Tensor]:
    """Generate one posterior reconstruction per grader annotation."""
    reconstructions = []
    for grader_idx in range(grader_masks.shape[1]):
        grader_mask = grader_masks[:, grader_idx : grader_idx + 1, :, :].float()
        logits, _ = model(x=image, y_target=grader_mask, sample_posterior=True)
        reconstructions.append(logits)
    return reconstructions


@torch.no_grad()
def generate_prior_samples(
    model: sPUNet, image: torch.Tensor, num_samples: int = 24
) -> List[torch.Tensor]:
    """Generate unconditional prior samples."""
    samples = []
    for _ in range(num_samples):
        logits, _ = model(x=image, y_target=None, sample_posterior=False)
        samples.append(logits)
    return samples


# ----------------------------
# Array conversion helpers
# ----------------------------

def tensor_to_numpy_mask(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert logits/prob/prob-like mask to binarized numpy (0/1)."""
    t = tensor.detach()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)
    arr = t.cpu().numpy().astype(np.float32)
    # If it looks like logits, apply sigmoid
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    return (arr > threshold).astype(np.uint8)


def tensor_to_numpy_ct_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CT tensor to numpy WITHOUT masking; rescale to [0,1] for display."""
    t = tensor.detach()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)
    arr = t.cpu().numpy().astype(np.float32)
    vmin, vmax = float(arr.min()), float(arr.max())
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr)
    return arr


# ----------------------------
# IoU / GED^2 metrics
# ----------------------------

def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a > 0, b > 0).sum(dtype=np.float64)
    union = np.logical_or(a > 0, b > 0).sum(dtype=np.float64)
    if union == 0.0:
        return 1.0  # both empty -> perfect agreement
    return float(inter / union)


def pairwise_iou(set_A: List[np.ndarray], set_B: List[np.ndarray]) -> np.ndarray:
    mat = np.zeros((len(set_A), len(set_B)), dtype=np.float64)
    for i, a in enumerate(set_A):
        for j, b in enumerate(set_B):
            mat[i, j] = iou_binary(a, b)
    return mat


def ged2_iou(set_S: List[np.ndarray], set_Y: List[np.ndarray], clamp_nonneg: bool = True) -> Dict[str, float]:
    """GED^2 with IoU distance: d = 1 - IoU; expectations include diagonals (with-replacement).
    GED^2 = 2*E[d(S,Y)] - E[d(S,S')] - E[d(Y,Y')]
    """
    if len(set_S) == 0 or len(set_Y) == 0:
        return {"GED2": float("nan"), "E_IoU_SY": float("nan"), "E_IoU_YY": float("nan"), "E_IoU_SS": float("nan")}

    iou_SY = pairwise_iou(set_S, set_Y)
    E_IoU_SY = float(iou_SY.mean())

    iou_SS = pairwise_iou(set_S, set_S)
    E_IoU_SS = float(iou_SS.mean())

    iou_YY = pairwise_iou(set_Y, set_Y)
    E_IoU_YY = float(iou_YY.mean())

    E_d_SY = 1.0 - E_IoU_SY
    E_d_SS = 1.0 - E_IoU_SS
    E_d_YY = 1.0 - E_IoU_YY

    GED2 = 2.0 * E_d_SY - E_d_SS - E_d_YY
    if clamp_nonneg and np.isfinite(GED2) and GED2 < 0.0:
        GED2 = 0.0

    return {"GED2": float(GED2), "E_IoU_SY": E_IoU_SY, "E_IoU_YY": E_IoU_YY, "E_IoU_SS": E_IoU_SS}


# ----------------------------
# Visualization (right-aligned + aligned columns)
# ----------------------------

def _add_subplot(fig, nrows, ncols, idx, img, title, cmap="gray", title_fontsize=14):
    ax = fig.add_subplot(nrows, ncols, idx)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=title_fontsize, loc="right")
    ax.axis("off")
    return ax


def create_visualization_grid_style1(
    ct_scan: np.ndarray,
    grader_masks: np.ndarray,  # [4,H,W]
    reconstructions: List[np.ndarray],
    recon_iou_list: List[Optional[float]],
    samples: List[np.ndarray],
    save_path: Path,
    title_fontsize: int = 14,
    footer_fontsize: int = 12,
):
    fig = plt.figure(figsize=(16, 12))

    # Row 1: CT + graders (cols 1..5)
    _add_subplot(fig, 6, 8, 1, ct_scan, "CT", title_fontsize=title_fontsize)
    for i in range(min(4, grader_masks.shape[0])):
        _add_subplot(fig, 6, 8, i + 2, grader_masks[i], f"grader {i+1}", title_fontsize=title_fontsize)

    # Row 2: Reconstructions aligned under graders (cols 2..5)
    for i in range(min(4, len(reconstructions))):
        idx = 8 + (i + 2)  # -> 10..13
        ax = fig.add_subplot(6, 8, idx)
        ax.imshow(reconstructions[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"recon {i+1}", fontsize=title_fontsize, loc="right")
        ax.axis("off")
        iou_val = recon_iou_list[i]
        txt = "IoU=NA" if iou_val is None or np.isnan(iou_val) else f"IoU={iou_val:.3f}"
        ax.text(1.0, -0.08, txt, transform=ax.transAxes, ha="right", va="top", fontsize=footer_fontsize)

    # Rows 3-6: 24 samples
    for i in range(min(24, len(samples))):
        row = 2 + (i // 8)
        col = (i % 8) + 1
        subplot_idx = row * 8 + col + 1
        _add_subplot(fig, 6, 8, subplot_idx, samples[i], f"s{i+1}", title_fontsize=title_fontsize)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {save_path}")


def create_visualization_grid_style2(
    ct_scan: np.ndarray,
    grader_masks: np.ndarray,
    reconstructions: List[np.ndarray],
    recon_iou_list: List[Optional[float]],
    samples: List[np.ndarray],
    save_path: Path,
    title_fontsize: int = 14,
    footer_fontsize: int = 12,
):
    fig = plt.figure(figsize=(15, 8))

    def add_at(row, col, total_cols, img, title):
        ax = fig.add_subplot(3, total_cols, row * total_cols + col + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=title_fontsize, loc="right")
        ax.axis("off")
        return ax

    # Row 1: CT at col1; graders at cols 2..5
    add_at(0, 0, 6, ct_scan, "CT")
    for i in range(min(4, grader_masks.shape[0])):
        add_at(0, i + 1, 6, grader_masks[i], f"grader {i+1}")

    # Row 2: Reconstructions aligned under graders -> cols 2..5
    for i in range(min(4, len(reconstructions))):
        ax = add_at(1, i + 1, 6, reconstructions[i], f"{i+1}" if i > 0 else "i) Reconstructions\n1")
        iou_val = recon_iou_list[i]
        txt = "IoU=NA" if iou_val is None or np.isnan(iou_val) else f"IoU={iou_val:.3f}"
        ax.text(1.0, -0.08, txt, transform=ax.transAxes, ha="right", va="top", fontsize=footer_fontsize)

    # Row 3: Samples (cols 1..6)
    for i in range(min(6, len(samples))):
        title = f"{i+1}" if i > 0 else "ii) Samples\n1"
        add_at(2, i, 6, samples[i], title)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved visualization to: {save_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="sPUNet Evaluation and Visualization (Updated)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=Path, required=True, help="Training config file")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--csv-name", type=str, default="test.csv", help="CSV split to use (e.g., train.csv/test.csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("spu_evaluation_results"), help="Output directory")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to visualize")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of prior samples per example")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--eval-scope", type=str, choices=["all", "visualize_only"], default="all",
                        help="Compute metrics over ALL rows in CSV (default) or only the visualized subset.")
    parser.add_argument("--ged-clamp-nonneg", action="store_true", default=True,
                        help="Clamp GED^2 to be non-negative (helps when small-sample bias yields tiny negatives).")

    args = parser.parse_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load config and model
    _ = load_config(args.config)
    model = load_model(args.checkpoint, device)

    # Dataset
    csv_path = args.data_root / args.csv_name
    ds = LIDCCropsDataset(
        csv_path=csv_path,
        project_root=args.data_root.parent.parent,
        image_size=128,
        augment=False,
        seed=42,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)
    print(f"Loaded {len(ds)} rows from {csv_path.name}")

    # Metrics accumulators
    rows_all: List[dict] = []
    rows_4g: List[dict] = []

    num_plotted = 0

    for row_idx, batch in enumerate(loader):
        if args.eval_scope == "visualize_only" and num_plotted >= args.num_examples:
            break

        image = batch["image"].to(device)   # [1,1,H,W]
        masks = batch["masks"].to(device)   # [1,4,H,W]

        reconstructions = generate_posterior_reconstructions(model, image, masks)
        samples = generate_prior_samples(model, image, args.num_samples)

        ct_scan = tensor_to_numpy_ct_image(image)
        grader_masks_np = [tensor_to_numpy_mask(masks[:, i : i + 1, :, :]) for i in range(masks.shape[1])]
        reconstructions_np = [tensor_to_numpy_mask(recon) for recon in reconstructions]
        samples_np = [tensor_to_numpy_mask(sample) for sample in samples]

        grader_available = [int(m.sum() > 0) for m in grader_masks_np]
        num_graders_avail = sum(grader_available)

        # GED^2 metrics
        Y_all = [m for m in grader_masks_np if m.sum() > 0]
        S_recon = reconstructions_np
        if len(Y_all) > 0 and len(S_recon) > 0:
            metrics_all = ged2_iou(S_recon, Y_all, clamp_nonneg=args.ged_clamp_nonneg)
        else:
            metrics_all = {"GED2": np.nan, "E_IoU_SY": np.nan, "E_IoU_YY": np.nan, "E_IoU_SS": np.nan}

        rows_all.append({
            "row_idx": row_idx,
            "num_graders": num_graders_avail,
            "GED2_IoU": metrics_all["GED2"],
            "E_IoU_SY": metrics_all["E_IoU_SY"],
            "E_IoU_YY": metrics_all["E_IoU_YY"],
            "E_IoU_SS": metrics_all["E_IoU_SS"],
        })

        if num_graders_avail == 4 and len(S_recon) > 0:
            metrics_4g = ged2_iou(S_recon, grader_masks_np, clamp_nonneg=args.ged_clamp_nonneg)
            rows_4g.append({
                "row_idx": row_idx,
                "num_graders": num_graders_avail,
                "GED2_IoU": metrics_4g["GED2"],
                "E_IoU_SY": metrics_4g["E_IoU_SY"],
                "E_IoU_YY": metrics_4g["E_IoU_YY"],
                "E_IoU_SS": metrics_4g["E_IoU_SS"],
            })

        # Visualizations (first N rows)
        if num_plotted < args.num_examples:
            H, W = reconstructions_np[0].shape
            blank = np.zeros((H, W), dtype=np.uint8)
            recon_iou_list = []
            for i in range(min(4, len(reconstructions_np))):
                ref = grader_masks_np[i] if grader_available[i] else blank
                recon_iou_list.append(iou_binary(reconstructions_np[i], ref))

            save_path_1 = args.output_dir / f"row_{row_idx:05d}_style1.png"
            create_visualization_grid_style1(
                ct_scan, np.array(grader_masks_np), reconstructions_np, recon_iou_list, samples_np,
                save_path_1, title_fontsize=14, footer_fontsize=12
            )

            save_path_2 = args.output_dir / f"row_{row_idx:05d}_style2.png"
            create_visualization_grid_style2(
                ct_scan, np.array(grader_masks_np), reconstructions_np, recon_iou_list, samples_np[:6],
                save_path_2, title_fontsize=14, footer_fontsize=12
            )

            num_plotted += 1

        if args.eval_scope == "visualize_only" and num_plotted >= args.num_examples:
            break

    # Save CSVs
    def write_csv(path: Path, rows: List[dict]):
        if not rows:
            print(f"No rows to write for {path.name}")
            return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"Wrote: {path}")

    write_csv(args.output_dir / "metrics_all_cases.csv", rows_all)
    write_csv(args.output_dir / "metrics_all4graders_only.csv", rows_4g)

    # Aggregates
    def summarize(rows: List[dict], label: str):
        if not rows:
            print(f"[{label}] No rows.")
            return
        import numpy as _np
        geds = _np.array([r["GED2_IoU"] for r in rows], dtype=_np.float64)
        e_sy = _np.array([r["E_IoU_SY"] for r in rows], dtype=_np.float64)
        print(f"[{label}] N={len(rows)} | GED2(mean±std)={_np.nanmean(geds):.4f}±{_np.nanstd(geds):.4f} | E[IoU(S,Y)]={_np.nanmean(e_sy):.4f}")

    summarize(rows_all, "ALL CASES (per CSV scope)")
    summarize(rows_4g, "ONLY 4 GRADERS")

    print(f"\nDone. sPUNet results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
