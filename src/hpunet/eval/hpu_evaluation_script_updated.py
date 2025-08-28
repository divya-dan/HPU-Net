#!/usr/bin/env python3
# HPU-Net Evaluation Script (Updated v2)
# - CT shown raw (no masking).
# - IoU-based GED² for reconstructions vs graders.
# - Metrics scope configurable: all rows in CSV (default) or only the visualized subset.
# - IoU shown *under each reconstruction* (recon_i vs grader_i).
# - All subplot titles/text right-aligned; larger fonts.
# - Saves per-case metrics (all cases + only-4-graders).

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
from hpunet.models.hpu_net import HPUNet
from hpunet.utils.config import load_config


# ----------------------------
# Model loading
# ----------------------------
def load_model(ckpt_path: Path, device: torch.device) -> HPUNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = HPUNet(in_ch=1, base=24, z_ch=1, n_blocks=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model from step {ckpt.get('step', 'unknown')}")
    return model


# ----------------------------
# Sampling helpers
# ----------------------------
@torch.no_grad()
def generate_posterior_reconstructions(
    model: HPUNet,
    image: torch.Tensor,
    grader_masks: torch.Tensor,
) -> List[torch.Tensor]:
    reconstructions = []
    for grader_idx in range(grader_masks.shape[1]):
        grader_mask = grader_masks[:, grader_idx : grader_idx + 1, :, :].float()
        logits, _ = model(x=image, y_target=grader_mask, sample_posterior=True)
        reconstructions.append(logits)
    return reconstructions


@torch.no_grad()
def generate_prior_samples(
    model: HPUNet, image: torch.Tensor, num_samples: int = 24
) -> List[torch.Tensor]:
    samples = []
    for _ in range(num_samples):
        logits, _ = model(x=image, y_target=None, sample_posterior=False)
        samples.append(logits)
    return samples


# ----------------------------
# Array conversion helpers
# ----------------------------
def tensor_to_numpy_mask(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
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
        return 1.0
    return float(inter / union)


def pairwise_iou(set_A: List[np.ndarray], set_B: List[np.ndarray]) -> np.ndarray:
    mat = np.zeros((len(set_A), len(set_B)), dtype=np.float64)
    for i, a in enumerate(set_A):
        for j, b in enumerate(set_B):
            mat[i, j] = iou_binary(a, b)
    return mat


def ged2_iou(set_S: List[np.ndarray], set_Y: List[np.ndarray]) -> Dict[str, float]:
    if len(set_S) == 0 or len(set_Y) == 0:
        return {"GED2": float("nan"), "E_IoU_SY": float("nan"), "E_IoU_YY": float("nan"), "E_IoU_SS": float("nan")}

    iou_SY = pairwise_iou(set_S, set_Y)
    E_IoU_SY = float(iou_SY.mean())

    if len(set_S) > 1:
        iou_SS = pairwise_iou(set_S, set_S)
        SS_no_diag = iou_SS.copy()
        np.fill_diagonal(SS_no_diag, np.nan)
        E_IoU_SS = float(np.nanmean(SS_no_diag))
    else:
        E_IoU_SS = 1.0

    if len(set_Y) > 1:
        iou_YY = pairwise_iou(set_Y, set_Y)
        YY_no_diag = iou_YY.copy()
        np.fill_diagonal(YY_no_diag, np.nan)
        E_IoU_YY = float(np.nanmean(YY_no_diag))
    else:
        E_IoU_YY = 1.0

    E_d_SY = 1.0 - E_IoU_SY
    E_d_SS = 1.0 - E_IoU_SS
    E_d_YY = 1.0 - E_IoU_YY

    GED2 = 2.0 * E_d_SY - E_d_SS - E_d_YY
    return {"GED2": float(GED2), "E_IoU_SY": E_IoU_SY, "E_IoU_YY": E_IoU_YY, "E_IoU_SS": E_IoU_SS}


# ----------------------------
# Visualization (right-aligned)
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
    recon_iou_list: List[Optional[float]],  # IoU per recon vs corresponding grader
    samples: List[np.ndarray],
    save_path: Path,
    title_fontsize: int = 14,
    footer_fontsize: int = 12,
):
    fig = plt.figure(figsize=(16, 12))

    # Row 1
    _add_subplot(fig, 6, 8, 1, ct_scan, "CT", title_fontsize=title_fontsize)
    for i in range(min(4, grader_masks.shape[0])):
        _add_subplot(fig, 6, 8, i + 2, grader_masks[i], f"grader {i+1}", title_fontsize=title_fontsize)

    # Row 2: Reconstructions with IoU footer (right-aligned)
    for i in range(min(4, len(reconstructions))):
        ax = fig.add_subplot(6, 8, 9 + i)
        ax.imshow(reconstructions[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"recon {i+1}", fontsize=title_fontsize, loc="right")
        ax.axis("off")
        iou_val = recon_iou_list[i]
        txt = "IoU=NA" if iou_val is None or np.isnan(iou_val) else f"IoU={iou_val:.3f}"
        # place under the image, right-aligned
        ax.text(1.0, -0.08, txt, transform=ax.transAxes, ha="right", va="top", fontsize=footer_fontsize)

    # Rows 3-6: Samples
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
    grader_masks: np.ndarray,  # [4,H,W]
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

    # Row 1
    add_at(0, 0, 6, ct_scan, "CT")
    for i in range(min(4, grader_masks.shape[0])):
        add_at(0, i + 1, 6, grader_masks[i], f"grader {i+1}")

    # Row 2: Reconstructions (with IoU footer)
    for i in range(min(4, len(reconstructions))):
        title = f"{i+1}" if i > 0 else "i) Reconstructions\n1"
        ax = add_at(1, i, 6, reconstructions[i], title)
        iou_val = recon_iou_list[i]
        txt = "IoU=NA" if iou_val is None or np.isnan(iou_val) else f"IoU={iou_val:.3f}"
        ax.text(1.0, -0.08, txt, transform=ax.transAxes, ha="right", va="top", fontsize=footer_fontsize)

    # Row 3: Samples
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
    parser = argparse.ArgumentParser(description="HPU-Net Evaluation and Visualization (Updated v2)")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=Path, required=True, help="Training config file")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--csv-name", type=str, default="test.csv", help="Which CSV to use (e.g., train.csv/test.csv)")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"), help="Output directory")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of examples to visualize")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of prior samples per example")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")
    parser.add_argument("--eval-scope", type=str, choices=["all", "visualize_only"], default="all",
                        help="Compute metrics over ALL rows in CSV (default) or only the visualized subset.")

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
    rows_all = []
    rows_4g = []

    # For visualization, we only save plots for first num-examples rows encountered
    num_plotted = 0

    for example_idx, batch in enumerate(loader):
        # If computing metrics only for visualized subset
        if args.eval_scope == "visualize_only" and num_plotted >= args.num_examples:
            break

        image = batch["image"].to(device)   # [1,1,H,W]
        masks = batch["masks"].to(device)   # [1,4,H,W]

        # Posterior reconstructions & prior samples
        reconstructions = generate_posterior_reconstructions(model, image, masks)
        samples = generate_prior_samples(model, image, args.num_samples)

        # Arrays
        ct_scan = tensor_to_numpy_ct_image(image)  # RAW CT
        grader_masks_np = [tensor_to_numpy_mask(masks[:, i : i + 1, :, :]) for i in range(masks.shape[1])]
        reconstructions_np = [tensor_to_numpy_mask(recon) for recon in reconstructions]
        samples_np = [tensor_to_numpy_mask(sample) for sample in samples]

        # Available graders
        grader_available = [int(m.sum() > 0) for m in grader_masks_np]
        num_graders_avail = sum(grader_available)

        # GED² metrics (recons vs ALL available graders)
        Y_all = [m for m in grader_masks_np if m.sum() > 0]
        S_recon = reconstructions_np
        metrics_all = ged2_iou(S_recon, Y_all)
        rows_all.append({
            "row_idx": example_idx,
            "num_graders": num_graders_avail,
            "GED2_IoU": metrics_all["GED2"],
            "E_IoU_SY": metrics_all["E_IoU_SY"],
            "E_IoU_YY": metrics_all["E_IoU_YY"],
            "E_IoU_SS": metrics_all["E_IoU_SS"],
        })

        # Only cases with all 4 graders
        if num_graders_avail == 4:
            metrics_4g = ged2_iou(S_recon, grader_masks_np)
            rows_4g.append({
                "row_idx": example_idx,
                "num_graders": num_graders_avail,
                "GED2_IoU": metrics_4g["GED2"],
                "E_IoU_SY": metrics_4g["E_IoU_SY"],
                "E_IoU_YY": metrics_4g["E_IoU_YY"],
                "E_IoU_SS": metrics_4g["E_IoU_SS"],
            })

        # Visualization for the first N examples only
        if num_plotted < args.num_examples:
            # IoU under each reconstruction: recon_i vs grader_i (if available; else NA)
            recon_iou_list = []
            for i in range(min(4, len(reconstructions_np))):
                if grader_available[i]:
                    recon_iou_list.append(iou_binary(reconstructions_np[i], grader_masks_np[i]))
                else:
                    recon_iou_list.append(np.nan)

            save_path_1 = args.output_dir / f"row_{example_idx:05d}_style1.png"
            create_visualization_grid_style1(
                ct_scan, np.array(grader_masks_np), reconstructions_np, recon_iou_list, samples_np,
                save_path_1, title_fontsize=14, footer_fontsize=12
            )

            save_path_2 = args.output_dir / f"row_{example_idx:05d}_style2.png"
            create_visualization_grid_style2(
                ct_scan, np.array(grader_masks_np), reconstructions_np, recon_iou_list, samples_np[:6],
                save_path_2, title_fontsize=14, footer_fontsize=12
            )

            num_plotted += 1

        # If metrics should span all rows, continue; otherwise we may break after plotted
        if args.eval_scope == "visualize_only" and num_plotted >= args.num_examples:
            break

    # Save metrics
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

    print(f"\nDone. Plots & CSVs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
