#!/usr/bin/env python3
# HPU-Net Evaluation Script (Updated v3)
# - CT shown raw (no masking).
# - IoU-based GED² for reconstructions vs graders.
# - Metrics scope configurable: all rows in CSV (default) or only the visualized subset.
# - IoU shown *under each reconstruction* (recon_i vs grader_i). If grader_i is empty/missing
#   and recon_i is also empty, IoU=1 (both blank).
# - All subplot titles/text right-aligned; larger fonts.
# - Grader and reconstruction columns are aligned.
# - GED² uses "with replacement" expectations (includes diagonal) and clamped to >= 0.
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


def load_model(ckpt_path: Path, device: torch.device) -> HPUNet:
    ckpt = torch.load(ckpt_path, map_location=device)
    model = HPUNet(in_ch=1, base=24, z_ch=1, n_blocks=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


@torch.no_grad()
def generate_posterior_reconstructions(model: HPUNet, image: torch.Tensor, grader_masks: torch.Tensor) -> List[torch.Tensor]:
    reconstructions = []
    for grader_idx in range(grader_masks.shape[1]):
        grader_mask = grader_masks[:, grader_idx : grader_idx + 1, :, :].float()
        logits, _ = model(x=image, y_target=grader_mask, sample_posterior=True)
        reconstructions.append(logits)
    return reconstructions


@torch.no_grad()
def generate_prior_samples(model: HPUNet, image: torch.Tensor, num_samples: int = 24) -> List[torch.Tensor]:
    samples = []
    for _ in range(num_samples):
        logits, _ = model(x=image, y_target=None, sample_posterior=False)
        samples.append(logits)
    return samples


def tensor_to_numpy_mask(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    t = tensor.detach()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)
    arr = t.cpu().numpy().astype(np.float32)
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
    E_IoU_SS = float(pairwise_iou(set_S, set_S).mean())
    E_IoU_YY = float(pairwise_iou(set_Y, set_Y).mean())
    GED2 = 2.0 * (1.0 - E_IoU_SY) - (1.0 - E_IoU_SS) - (1.0 - E_IoU_YY)
    GED2 = max(0.0, GED2)
    return {"GED2": GED2, "E_IoU_SY": E_IoU_SY, "E_IoU_YY": E_IoU_YY, "E_IoU_SS": E_IoU_SS}


def _add_subplot(fig, nrows, ncols, idx, img, title, cmap="gray", title_fontsize=14):
    ax = fig.add_subplot(nrows, ncols, idx)
    ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
    ax.set_title(title, fontsize=title_fontsize, loc="right")
    ax.axis("off")
    return ax


def create_visualization_grid(ct_scan, grader_masks, reconstructions, recon_iou_list, samples, save_path,
                           title_fontsize: int = 14, footer_fontsize: int = 12):
    """
    3 x 6 grid with **column-aligned** graders and reconstructions:
      Row 1 (cols 1..6): [CT] [grader1] [grader2] [grader3] [grader4] [empty]
      Row 2 (cols 1..6): [empty] [recon1] [recon2] [recon3] [recon4] [empty]
      Row 3 (cols 1..6): [sample1..6]
    All titles right-aligned; IoU printed under each recon panel (right-aligned).
    """
    fig = plt.figure(figsize=(16, 12))

    def add_at(row, col, img, title):
        ax = fig.add_subplot(3, 6, row * 6 + col + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=title_fontsize, loc="right")
        ax.axis("off")
        return ax

    # Row 1: CT + graders in columns 1..5
    add_at(0, 0, ct_scan, "CT")
    for i in range(min(4, grader_masks.shape[0])):
        add_at(0, i + 1, grader_masks[i], f"grader {i+1}")

    # Row 2: Reconstructions aligned directly under the corresponding graders -> cols 2..5
    for i in range(min(4, len(reconstructions))):
        ax = add_at(1, i + 1, reconstructions[i], f"recon {i+1}")
        iou_val = recon_iou_list[i]
        txt = "IoU=NA" if iou_val is None or np.isnan(iou_val) else f"IoU={iou_val:.3f}"
        ax.text(1.0, -0.08, txt, transform=ax.transAxes, ha="right", va="top", fontsize=footer_fontsize)

    # Row 3: Samples in cols 1..6
    for i in range(min(6, len(samples))):
        add_at(2, i, samples[i], f"s{i+1}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="HPU-Net Evaluation and Visualization (Updated v3)")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--csv-name", type=str, default="test.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation_results"))
    parser.add_argument("--num-examples", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=24)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval-scope", type=str, choices=["all", "visualize_only"], default="all")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _ = load_config(args.config)
    model = load_model(args.checkpoint, device)
    csv_path = args.data_root / args.csv_name
    ds = LIDCCropsDataset(csv_path=csv_path, project_root=args.data_root.parent.parent, image_size=128, augment=False, seed=42)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)

    rows_all, rows_4g = [], []
    num_plotted = 0
    for row_idx, batch in enumerate(loader):
        if args.eval_scope == "visualize_only" and num_plotted >= args.num_examples:
            break
        image, masks = batch["image"].to(device), batch["masks"].to(device)
        reconstructions = generate_posterior_reconstructions(model, image, masks)
        samples = generate_prior_samples(model, image, args.num_samples)
        ct_scan = tensor_to_numpy_ct_image(image)
        grader_masks_np = [tensor_to_numpy_mask(masks[:, i:i+1]) for i in range(masks.shape[1])]
        reconstructions_np = [tensor_to_numpy_mask(r) for r in reconstructions]
        samples_np = [tensor_to_numpy_mask(s) for s in samples]
        grader_available = [int(m.sum() > 0) for m in grader_masks_np]
        num_graders_avail = sum(grader_available)
        Y_all = [m for m in grader_masks_np if m.sum() > 0]
        metrics_all = ged2_iou(reconstructions_np, Y_all) if Y_all else {"GED2": np.nan, "E_IoU_SY": np.nan, "E_IoU_YY": np.nan, "E_IoU_SS": np.nan}
        rows_all.append({"row_idx": row_idx, "num_graders": num_graders_avail, **metrics_all})
        if num_graders_avail == 4:
            rows_4g.append({"row_idx": row_idx, "num_graders": num_graders_avail, **ged2_iou(reconstructions_np, grader_masks_np)})
        if num_plotted < args.num_examples:
            blank = np.zeros_like(reconstructions_np[0])
            recon_iou_list = [iou_binary(reconstructions_np[i], grader_masks_np[i] if grader_available[i] else blank) for i in range(4)]
            save_path = args.output_dir / f"row_{row_idx:05d}.png"
            create_visualization_grid(ct_scan, np.array(grader_masks_np), reconstructions_np, recon_iou_list, samples_np, save_path)
            num_plotted += 1

    def write_csv(path: Path, rows: List[dict]):
        if not rows: return
        keys = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader(); w.writerows(rows)
    write_csv(args.output_dir / "metrics_all_cases.csv", rows_all)
    write_csv(args.output_dir / "metrics_all4graders_only.csv", rows_4g)


if __name__ == "__main__":
    main()
