#!/usr/bin/env python3
"""
Comparative Evaluation Notebook: sPUNet vs HPUNet
Computes IoU and GED² metrics with modified handling of empty ground truth cases:
- Case 1: GT empty + pred empty → IoU = 1 (but excluded from aggregate stats)
- Case 2: GT empty → exclude from evaluation entirely
"""
# ============================================================================
# 1. IMPORTS & UTILITIES
# ============================================================================

from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import csv

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# Import your model classes and utilities
from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import sPUNet
from hpunet.models.hpu_net import HPUNet
from hpunet.utils.config import load_config


# ============================================================================
# 2. CONFIGURATION & SETUP
# ============================================================================

# Model and data paths (UPDATE THESE FOR YOUR SYSTEM)
SPUNET_CHECKPOINT = "runs/runs/iwi9140h-project/spu/run_20250829_082129_1197064/spu_last.pth"  # Update this path
SPUNET_CONFIG = "runs/runs/iwi9140h-project/spu/run_20250829_082129_1197064/train_spu_lidc.json"         # Update this path
HPUNET_CHECKPOINT = "runs/runs/iwi9140h-project/hpu/run_20250829_082159_1197065/hpu_last.pth"  # Update this path  
HPUNET_CONFIG = "runs/runs/iwi9140h-project/hpu/run_20250829_082159_1197065/train_hpu_lidc.json"         # Update this path

DATA_ROOT = "data/lidc_crops"                      # Update this path
CSV_NAME = "test.csv"                                # or "train.csv"

# Evaluation parameters
OUTPUT_DIR = "runs/runs/iwi9140h-project/comparative_evaluation_results"
NUM_EXAMPLES = 5          # Number of cases to visualize
NUM_SAMPLES = 8          # Prior samples per model per case
DEVICE = "cuda"           # or "cpu"
EVAL_SCOPE = "all"        # "all" or "visualize_only"
GED_CLAMP_NONNEG = True   # Clamp GED² to non-negative

# Visualization settings
TITLE_FONTSIZE = 14
FOOTER_FONTSIZE = 12
FIGURE_DPI = 150

print("Configuration loaded. Update paths above before running subsequent cells.")

# Setup
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
output_dir = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Using device: {device}")
print(f"Output directory: {output_dir}")

# ============================================================================
# 3. UTILITY FUNCTIONS
# ============================================================================

def tensor_to_numpy_mask(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert logits/prob tensor to binary numpy mask (0/1)."""
    t = tensor.detach()
    if t.dim() == 4:
        t = t.squeeze(0).squeeze(0)
    elif t.dim() == 3:
        t = t.squeeze(0)
    arr = t.cpu().numpy().astype(np.float32)
    
    # Apply sigmoid if it looks like logits
    if arr.max() > 1.0 or arr.min() < 0.0:
        arr = 1.0 / (1.0 + np.exp(-arr))
    
    return (arr > threshold).astype(np.uint8)


def tensor_to_numpy_ct_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CT tensor to numpy, rescaled to [0,1] for display."""
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


def is_empty_mask(mask: np.ndarray) -> bool:
    """Check if mask is completely empty (all pixels = 0)."""
    return mask.sum() == 0


def iou_binary(a: np.ndarray, b: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    inter = np.logical_and(a > 0, b > 0).sum(dtype=np.float64)
    union = np.logical_or(a > 0, b > 0).sum(dtype=np.float64)
    
    if union == 0.0:
        return 1.0  # Both empty -> perfect agreement
    
    return float(inter / union)


def pairwise_iou(set_A: List[np.ndarray], set_B: List[np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU matrix between two sets of masks."""
    mat = np.zeros((len(set_A), len(set_B)), dtype=np.float64)
    for i, a in enumerate(set_A):
        for j, b in enumerate(set_B):
            mat[i, j] = iou_binary(a, b)
    return mat


def ged2_iou(set_S: List[np.ndarray], set_Y: List[np.ndarray], clamp_nonneg: bool = True) -> Dict[str, float]:
    """
    Compute GED² using IoU-based distance.
    GED² = 2*E[d(S,Y)] - E[d(S,S')] - E[d(Y,Y')]
    where d = 1 - IoU
    """
    if len(set_S) == 0 or len(set_Y) == 0:
        return {
            "GED2": float("nan"), 
            "E_IoU_SY": float("nan"), 
            "E_IoU_YY": float("nan"), 
            "E_IoU_SS": float("nan")
        }

    # Compute pairwise IoU matrices
    iou_SY = pairwise_iou(set_S, set_Y)
    iou_SS = pairwise_iou(set_S, set_S)
    iou_YY = pairwise_iou(set_Y, set_Y)
    
    # Compute expectations (includes diagonal terms)
    E_IoU_SY = float(iou_SY.mean())
    E_IoU_SS = float(iou_SS.mean())
    E_IoU_YY = float(iou_YY.mean())
    
    # Convert to distance and compute GED²
    E_d_SY = 1.0 - E_IoU_SY
    E_d_SS = 1.0 - E_IoU_SS
    E_d_YY = 1.0 - E_IoU_YY
    
    GED2 = 2.0 * E_d_SY - E_d_SS - E_d_YY
    
    if clamp_nonneg and np.isfinite(GED2) and GED2 < 0.0:
        GED2 = 0.0

    return {
        "GED2": float(GED2), 
        "E_IoU_SY": E_IoU_SY, 
        "E_IoU_YY": E_IoU_YY, 
        "E_IoU_SS": E_IoU_SS
    }


print("Utility functions loaded.")

# ============================================================================
# 4. MODEL LOADING
# ============================================================================

def load_spunet(ckpt_path: Path, device: torch.device) -> sPUNet:
    """Load sPUNet model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = sPUNet(in_ch=1, base=32, z_dim=6).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded sPUNet from step {ckpt.get('step', 'unknown')}")
    return model


def load_hpunet(ckpt_path: Path, device: torch.device) -> HPUNet:
    """Load HPUNet model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location=device)
    model = HPUNet(in_ch=1, base=24, z_ch=1, n_blocks=3).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded HPUNet from step {ckpt.get('step', 'unknown')}")
    return model


# Load both models
print("Loading models...")
spu_model = load_spunet(Path(SPUNET_CHECKPOINT), device)
hpu_model = load_hpunet(Path(HPUNET_CHECKPOINT), device)

# Load configs (if needed)
spu_config = load_config(Path(SPUNET_CONFIG))
hpu_config = load_config(Path(HPUNET_CONFIG))

print("Both models loaded successfully!")

# ============================================================================
# 5. DATA LOADING
# ============================================================================

# Setup dataset and dataloader
csv_path = Path(DATA_ROOT) / CSV_NAME
dataset = LIDCCropsDataset(
    csv_path=csv_path,
    project_root=Path(DATA_ROOT).parent.parent,
    image_size=128,
    augment=False,
    seed=42,
)

dataloader = DataLoader(
    dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=2
)

print(f"Loaded dataset with {len(dataset)} samples from {csv_path.name}")

# ============================================================================
# 6. INFERENCE FUNCTIONS
# ============================================================================

@torch.no_grad()
def generate_reconstructions(model, image: torch.Tensor, grader_masks: torch.Tensor) -> List[torch.Tensor]:
    """Generate posterior reconstructions for any model type."""
    reconstructions = []
    for grader_idx in range(grader_masks.shape[1]):
        grader_mask = grader_masks[:, grader_idx : grader_idx + 1, :, :].float()
        logits, _ = model(x=image, y_target=grader_mask, sample_posterior=True)
        reconstructions.append(logits)
    return reconstructions


@torch.no_grad()
def generate_samples(model, image: torch.Tensor, num_samples: int = 24) -> List[torch.Tensor]:
    """Generate prior samples for any model type."""
    samples = []
    for _ in range(num_samples):
        logits, _ = model(x=image, y_target=None, sample_posterior=False)
        samples.append(logits)
    return samples


print("Inference functions ready.")

# ============================================================================
# 7. COMPARATIVE METRICS COMPUTATION
# ============================================================================

def compute_comparative_metrics_all_cases(
    spu_reconstructions: List[np.ndarray],
    hpu_reconstructions: List[np.ndarray], 
    grader_masks: List[np.ndarray],
    clamp_nonneg: bool = True
) -> Dict[str, float]:
    """
    Compute metrics for both models INCLUDING empty GT cases:
    - Case 1: GT empty + pred empty → IoU = 1 (INCLUDED in stats)
    - This measures overall performance including "no lesion" prediction ability
    """
    
    if len(grader_masks) == 0:
        return {
            "num_graders": 0,
            "spu_GED2_all": float("nan"), "spu_E_IoU_SY_all": float("nan"),
            "hpu_GED2_all": float("nan"), "hpu_E_IoU_SY_all": float("nan"),
            "spu_E_IoU_SS_all": float("nan"), "spu_E_IoU_YY_all": float("nan"),
            "hpu_E_IoU_SS_all": float("nan"), "hpu_E_IoU_YY_all": float("nan")
        }
    
    # Compute metrics for sPUNet (all graders included)
    spu_metrics = ged2_iou(spu_reconstructions, grader_masks, clamp_nonneg)
    
    # Compute metrics for HPUNet (all graders included)
    hpu_metrics = ged2_iou(hpu_reconstructions, grader_masks, clamp_nonneg)
    
    return {
        "num_graders": len(grader_masks),
        "spu_GED2_all": spu_metrics["GED2"],
        "spu_E_IoU_SY_all": spu_metrics["E_IoU_SY"], 
        "spu_E_IoU_SS_all": spu_metrics["E_IoU_SS"],
        "spu_E_IoU_YY_all": spu_metrics["E_IoU_YY"],
        "hpu_GED2_all": hpu_metrics["GED2"],
        "hpu_E_IoU_SY_all": hpu_metrics["E_IoU_SY"],
        "hpu_E_IoU_SS_all": hpu_metrics["E_IoU_SS"], 
        "hpu_E_IoU_YY_all": hpu_metrics["E_IoU_YY"]
    }


def compute_comparative_metrics_nonempty_only(
    spu_reconstructions: List[np.ndarray],
    hpu_reconstructions: List[np.ndarray], 
    grader_masks: List[np.ndarray],
    clamp_nonneg: bool = True
) -> Dict[str, float]:
    """
    Compute metrics for both models EXCLUDING empty GT cases:
    - Case 2: Only evaluate on non-empty ground truth
    - This measures lesion detection performance when lesions are present
    """
    
    # Filter out empty ground truth masks and corresponding reconstructions
    non_empty_indices = [i for i, mask in enumerate(grader_masks) if not is_empty_mask(mask)]
    
    if len(non_empty_indices) == 0:
        return {
            "num_nonempty_graders": 0,
            "spu_GED2_nonempty": float("nan"), "spu_E_IoU_SY_nonempty": float("nan"),
            "hpu_GED2_nonempty": float("nan"), "hpu_E_IoU_SY_nonempty": float("nan"),
            "spu_E_IoU_SS_nonempty": float("nan"), "spu_E_IoU_YY_nonempty": float("nan"),
            "hpu_E_IoU_SS_nonempty": float("nan"), "hpu_E_IoU_YY_nonempty": float("nan")
        }
    
    # Get non-empty graders and corresponding reconstructions
    non_empty_graders = [grader_masks[i] for i in non_empty_indices]
    spu_recons_nonempty = [spu_reconstructions[i] for i in non_empty_indices]
    hpu_recons_nonempty = [hpu_reconstructions[i] for i in non_empty_indices]
    
    # Compute metrics for sPUNet (non-empty only)
    spu_metrics = ged2_iou(spu_recons_nonempty, non_empty_graders, clamp_nonneg)
    
    # Compute metrics for HPUNet (non-empty only)
    hpu_metrics = ged2_iou(hpu_recons_nonempty, non_empty_graders, clamp_nonneg)
    
    return {
        "num_nonempty_graders": len(non_empty_graders),
        "spu_GED2_nonempty": spu_metrics["GED2"],
        "spu_E_IoU_SY_nonempty": spu_metrics["E_IoU_SY"], 
        "spu_E_IoU_SS_nonempty": spu_metrics["E_IoU_SS"],
        "spu_E_IoU_YY_nonempty": spu_metrics["E_IoU_YY"],
        "hpu_GED2_nonempty": hpu_metrics["GED2"],
        "hpu_E_IoU_SY_nonempty": hpu_metrics["E_IoU_SY"],
        "hpu_E_IoU_SS_nonempty": hpu_metrics["E_IoU_SS"], 
        "hpu_E_IoU_YY_nonempty": hpu_metrics["E_IoU_YY"]
    }


def compute_individual_ious(
    reconstructions: List[np.ndarray], 
    grader_masks: List[np.ndarray]
) -> List[Optional[float]]:
    """
    Compute IoU between each reconstruction and corresponding grader.
    Always computes IoU (including empty cases for visualization).
    """
    ious = []
    for i in range(len(reconstructions)):
        if i < len(grader_masks):
            grader = grader_masks[i]
            recon = reconstructions[i]
            ious.append(iou_binary(recon, grader))  # Always compute IoU
        else:
            ious.append(None)
    
    return ious


print("Comparative metrics functions ready.")

# ============================================================================
# 8. VISUALIZATION FUNCTIONS
# ============================================================================

def create_comparative_visualization(
    ct_scan: np.ndarray,
    grader_masks: List[np.ndarray],
    spu_reconstructions: List[np.ndarray],
    hpu_reconstructions: List[np.ndarray],
    spu_ious: List[Optional[float]],
    hpu_ious: List[Optional[float]], 
    spu_samples: List[np.ndarray],
    hpu_samples: List[np.ndarray],
    save_path: Path
):
    """
    Create side-by-side comparison visualization:
    Row 1: [CT] [grader1] [grader2] [grader3] [grader4] [empty]
    Row 2: [empty] [sPU_recon1] [sPU_recon2] [sPU_recon3] [sPU_recon4] [empty]  
    Row 3: [empty] [HPU_recon1] [HPU_recon2] [HPU_recon3] [HPU_recon4] [empty]
    Row 4: [sPU_sample1-6]
    Row 5: [HPU_sample1-6]
    """
    fig = plt.figure(figsize=(18, 15))
    
    def add_subplot(row, col, img, title, show_iou=False, iou_val=None):
        ax = fig.add_subplot(5, 6, row * 6 + col + 1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=TITLE_FONTSIZE, loc="right")
        ax.axis("off")
        
        if show_iou:
            if iou_val is None:
                txt = "IoU=NA"
            elif np.isnan(iou_val):
                txt = "IoU=NA"
            else:
                txt = f"IoU={iou_val:.3f}"
            ax.text(1.0, -0.08, txt, transform=ax.transAxes, 
                   ha="right", va="top", fontsize=FOOTER_FONTSIZE)
        
        return ax
    
    # Row 1: CT + Graders
    add_subplot(0, 0, ct_scan, "CT")
    for i in range(min(4, len(grader_masks))):
        add_subplot(0, i + 1, grader_masks[i], f"grader {i+1}")
    
    # Row 2: sPUNet Reconstructions
    for i in range(min(4, len(spu_reconstructions))):
        title = f"sPU_recon {i+1}" if i > 0 else "sPUNet Reconstructions\nsPU_recon 1"
        add_subplot(1, i + 1, spu_reconstructions[i], title, show_iou=True, iou_val=spu_ious[i])
    
    # Row 3: HPUNet Reconstructions  
    for i in range(min(4, len(hpu_reconstructions))):
        title = f"HPU_recon {i+1}" if i > 0 else "HPUNet Reconstructions\nHPU_recon 1"
        add_subplot(2, i + 1, hpu_reconstructions[i], title, show_iou=True, iou_val=hpu_ious[i])
    
    # Row 4: sPUNet Samples
    for i in range(min(6, len(spu_samples))):
        title = f"sPU_s{i+1}" if i > 0 else "sPUNet Samples\nsPU_s1"
        add_subplot(3, i, spu_samples[i], title)
    
    # Row 5: HPUNet Samples
    for i in range(min(6, len(hpu_samples))):
        title = f"HPU_s{i+1}" if i > 0 else "HPUNet Samples\nHPU_s1"
        add_subplot(4, i, hpu_samples[i], title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved comparative visualization: {save_path}")


print("Visualization functions ready.")

# ============================================================================
# 9. MAIN COMPARATIVE EVALUATION
# ============================================================================

print("Starting comparative evaluation...")

# Metrics storage
all_case_metrics = []  # All cases in CSV scope
four_grader_metrics = []  # Only cases with all 4 graders available

num_visualized = 0

for row_idx, batch in enumerate(dataloader):
    if EVAL_SCOPE == "visualize_only" and num_visualized >= NUM_EXAMPLES:
        break
    
    print(f"Processing row {row_idx}...")
    
    # Get inputs
    image = batch["image"].to(device)  # [1,1,H,W]
    masks = batch["masks"].to(device)  # [1,4,H,W]
    
    # Generate predictions for both models
    spu_reconstructions = generate_reconstructions(spu_model, image, masks)
    hpu_reconstructions = generate_reconstructions(hpu_model, image, masks)
    
    spu_samples = generate_samples(spu_model, image, NUM_SAMPLES)
    hpu_samples = generate_samples(hpu_model, image, NUM_SAMPLES)
    
    # Convert to numpy
    ct_scan = tensor_to_numpy_ct_image(image)
    grader_masks_np = [tensor_to_numpy_mask(masks[:, i:i+1]) for i in range(masks.shape[1])]
    
    spu_reconstructions_np = [tensor_to_numpy_mask(r) for r in spu_reconstructions]
    hpu_reconstructions_np = [tensor_to_numpy_mask(r) for r in hpu_reconstructions]
    
    spu_samples_np = [tensor_to_numpy_mask(s) for s in spu_samples]
    hpu_samples_np = [tensor_to_numpy_mask(s) for s in hpu_samples]
    
    # Check grader availability (non-empty masks)
    grader_available = [not is_empty_mask(mask) for mask in grader_masks_np]
    num_available_graders = sum(grader_available)
    total_graders = len(grader_masks_np)
    
    # Compute metrics for CASE 1: All cases included (measures overall performance)
    metrics_all_included = compute_comparative_metrics_all_cases(
        spu_reconstructions_np, hpu_reconstructions_np, grader_masks_np, GED_CLAMP_NONNEG
    )
    
    # Compute metrics for CASE 2: Non-empty GT only (measures lesion detection performance)  
    metrics_nonempty_only = compute_comparative_metrics_nonempty_only(
        spu_reconstructions_np, hpu_reconstructions_np, grader_masks_np, GED_CLAMP_NONNEG
    )
    
    # Individual IoUs for visualization (always computed for display)
    spu_ious = compute_individual_ious(spu_reconstructions_np, grader_masks_np)
    hpu_ious = compute_individual_ious(hpu_reconstructions_np, grader_masks_np)
    
    # Store comprehensive metrics for all cases
    all_case_row = {
        "row_idx": row_idx,
        "total_graders": total_graders,
        "num_available_graders": num_available_graders,
        # Case 1: All cases included metrics
        **{k: v for k, v in metrics_all_included.items()},
        # Case 2: Non-empty only metrics  
        **{k: v for k, v in metrics_nonempty_only.items()}
    }
    all_case_metrics.append(all_case_row)
    
    # Store metrics for 4-grader cases only
    if total_graders == 4:
        four_grader_metrics.append(all_case_row.copy())
    
    # Create visualization for first N examples
    if num_visualized < NUM_EXAMPLES:
        save_path = output_dir / f"comparative_row_{row_idx:05d}.png"
        create_comparative_visualization(
            ct_scan, grader_masks_np, 
            spu_reconstructions_np, hpu_reconstructions_np,
            spu_ious, hpu_ious,
            spu_samples_np[:6], hpu_samples_np[:6],
            save_path
        )
        num_visualized += 1
    
    if EVAL_SCOPE == "visualize_only" and num_visualized >= NUM_EXAMPLES:
        break

print(f"Evaluation complete. Processed {len(all_case_metrics)} cases.")

# ============================================================================
# 10. RESULTS ANALYSIS & EXPORT
# ============================================================================

def save_metrics_csv(metrics_list: List[dict], filename: str):
    """Save metrics to CSV file."""
    if not metrics_list:
        print(f"No data to save for {filename}")
        return
        
    csv_path = output_dir / filename
    fieldnames = list(metrics_list[0].keys())
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_list)
    
    print(f"Saved: {csv_path}")


def print_summary_statistics(metrics_list: List[dict], label: str):
    """Print summary statistics for both evaluation cases."""
    if not metrics_list:
        print(f"[{label}] No data available.")
        return
    
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print(f"Total cases: {len(metrics_list)}")
    
    # Case 1: All cases included (overall performance including empty prediction)
    print(f"\n{'-'*50}")
    print(f"CASE 1: ALL CASES INCLUDED (Overall Performance)")
    print(f"{'-'*50}")
    
    spu_ged2_all = np.array([r["spu_GED2_all"] for r in metrics_list if not np.isnan(r["spu_GED2_all"])])
    hpu_ged2_all = np.array([r["hpu_GED2_all"] for r in metrics_list if not np.isnan(r["hpu_GED2_all"])])
    spu_iou_all = np.array([r["spu_E_IoU_SY_all"] for r in metrics_list if not np.isnan(r["spu_E_IoU_SY_all"])])
    hpu_iou_all = np.array([r["hpu_E_IoU_SY_all"] for r in metrics_list if not np.isnan(r["hpu_E_IoU_SY_all"])])
    
    print(f"Valid cases: sPUNet={len(spu_ged2_all)}, HPUNet={len(hpu_ged2_all)}")
    
    if len(spu_ged2_all) > 0:
        print(f"\nsPUNet (All Cases):")
        print(f"  GED² = {np.mean(spu_ged2_all):.4f} ± {np.std(spu_ged2_all):.4f}")
        print(f"  E[IoU(S,Y)] = {np.mean(spu_iou_all):.4f} ± {np.std(spu_iou_all):.4f}")
    
    if len(hpu_ged2_all) > 0:
        print(f"\nHPUNet (All Cases):")
        print(f"  GED² = {np.mean(hpu_ged2_all):.4f} ± {np.std(hpu_ged2_all):.4f}")
        print(f"  E[IoU(S,Y)] = {np.mean(hpu_iou_all):.4f} ± {np.std(hpu_iou_all):.4f}")
    
    if len(spu_ged2_all) > 0 and len(hpu_ged2_all) > 0:
        ged2_diff_all = np.mean(hpu_ged2_all) - np.mean(spu_ged2_all)
        iou_diff_all = np.mean(hpu_iou_all) - np.mean(spu_iou_all)
        print(f"\nComparison (HPUNet - sPUNet, All Cases):")
        print(f"  ΔGED² = {ged2_diff_all:+.4f} {'(HPUNet better)' if ged2_diff_all < 0 else '(sPUNet better)'}")
        print(f"  ΔE[IoU(S,Y)] = {iou_diff_all:+.4f} {'(HPUNet better)' if iou_diff_all > 0 else '(sPUNet better)'}")
    
    # Case 2: Non-empty GT only (lesion detection performance)
    print(f"\n{'-'*50}")  
    print(f"CASE 2: NON-EMPTY GT ONLY (Lesion Detection Performance)")
    print(f"{'-'*50}")
    
    spu_ged2_nonempty = np.array([r["spu_GED2_nonempty"] for r in metrics_list if not np.isnan(r["spu_GED2_nonempty"])])
    hpu_ged2_nonempty = np.array([r["hpu_GED2_nonempty"] for r in metrics_list if not np.isnan(r["hpu_GED2_nonempty"])])
    spu_iou_nonempty = np.array([r["spu_E_IoU_SY_nonempty"] for r in metrics_list if not np.isnan(r["spu_E_IoU_SY_nonempty"])])
    hpu_iou_nonempty = np.array([r["hpu_E_IoU_SY_nonempty"] for r in metrics_list if not np.isnan(r["hpu_E_IoU_SY_nonempty"])])
    
    print(f"Valid cases: sPUNet={len(spu_ged2_nonempty)}, HPUNet={len(hpu_ged2_nonempty)}")
    
    if len(spu_ged2_nonempty) > 0:
        print(f"\nsPUNet (Non-Empty GT Only):")
        print(f"  GED² = {np.mean(spu_ged2_nonempty):.4f} ± {np.std(spu_ged2_nonempty):.4f}")
        print(f"  E[IoU(S,Y)] = {np.mean(spu_iou_nonempty):.4f} ± {np.std(spu_iou_nonempty):.4f}")
    
    if len(hpu_ged2_nonempty) > 0:
        print(f"\nHPUNet (Non-Empty GT Only):")
        print(f"  GED² = {np.mean(hpu_ged2_nonempty):.4f} ± {np.std(hpu_ged2_nonempty):.4f}")
        print(f"  E[IoU(S,Y)] = {np.mean(hpu_iou_nonempty):.4f} ± {np.std(hpu_iou_nonempty):.4f}")
    
    if len(spu_ged2_nonempty) > 0 and len(hpu_ged2_nonempty) > 0:
        ged2_diff_nonempty = np.mean(hpu_ged2_nonempty) - np.mean(spu_ged2_nonempty)
        iou_diff_nonempty = np.mean(hpu_iou_nonempty) - np.mean(spu_iou_nonempty)
        print(f"\nComparison (HPUNet - sPUNet, Non-Empty Only):")
        print(f"  ΔGED² = {ged2_diff_nonempty:+.4f} {'(HPUNet better)' if ged2_diff_nonempty < 0 else '(sPUNet better)'}")
        print(f"  ΔE[IoU(S,Y)] = {iou_diff_nonempty:+.4f} {'(HPUNet better)' if iou_diff_nonempty > 0 else '(sPUNet better)'}")


# Save detailed metrics
save_metrics_csv(all_case_metrics, "comparative_metrics_all_cases.csv")
save_metrics_csv(four_grader_metrics, "comparative_metrics_4graders_only.csv")

# Print summary statistics
print_summary_statistics(all_case_metrics, "ALL CASES")
print_summary_statistics(four_grader_metrics, "ONLY 4-GRADER CASES")

# ============================================================================
# 11. FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("COMPARATIVE EVALUATION COMPLETE")
print(f"{'='*80}")
print(f"Dataset: {csv_path.name}")
print(f"Evaluation scope: {EVAL_SCOPE}")
print(f"Evaluation cases:")
print(f"  Case 1: All cases included (overall performance)")
print(f"  Case 2: Non-empty GT only (lesion detection performance)")
print(f"sPUNet checkpoint: {Path(SPUNET_CHECKPOINT).name}")
print(f"HPUNet checkpoint: {Path(HPUNET_CHECKPOINT).name}")
print(f"Results saved to: {output_dir}")
print(f"Visualizations created: {num_visualized}")
print(f"Total cases evaluated: {len(all_case_metrics)}")
print(f"Cases with 4 graders: {len(four_grader_metrics)}")

# Create comprehensive summary comparison table
summary_data = []
for label, metrics_list in [("All Cases", all_case_metrics), ("4-Grader Cases", four_grader_metrics)]:
    if not metrics_list:
        continue
    
    # Case 1: All cases metrics
    spu_ged2_all = [r["spu_GED2_all"] for r in metrics_list if not np.isnan(r["spu_GED2_all"])]
    hpu_ged2_all = [r["hpu_GED2_all"] for r in metrics_list if not np.isnan(r["hpu_GED2_all"])]
    spu_iou_all = [r["spu_E_IoU_SY_all"] for r in metrics_list if not np.isnan(r["spu_E_IoU_SY_all"])]
    hpu_iou_all = [r["hpu_E_IoU_SY_all"] for r in metrics_list if not np.isnan(r["hpu_E_IoU_SY_all"])]
    
    # Case 2: Non-empty only metrics
    spu_ged2_nonempty = [r["spu_GED2_nonempty"] for r in metrics_list if not np.isnan(r["spu_GED2_nonempty"])]
    hpu_ged2_nonempty = [r["hpu_GED2_nonempty"] for r in metrics_list if not np.isnan(r["hpu_GED2_nonempty"])]
    spu_iou_nonempty = [r["spu_E_IoU_SY_nonempty"] for r in metrics_list if not np.isnan(r["spu_E_IoU_SY_nonempty"])]
    hpu_iou_nonempty = [r["hpu_E_IoU_SY_nonempty"] for r in metrics_list if not np.isnan(r["hpu_E_IoU_SY_nonempty"])]
    
    if (spu_ged2_all and hpu_ged2_all and spu_iou_all and hpu_iou_all and 
        spu_ged2_nonempty and hpu_ged2_nonempty and spu_iou_nonempty and hpu_iou_nonempty):
        
        summary_data.extend([
            {
                "Dataset": f"{label}_AllCases",
                "Evaluation_Type": "All Cases Included", 
                "sPUNet_GED2_mean": np.mean(spu_ged2_all),
                "sPUNet_GED2_std": np.std(spu_ged2_all),
                "HPUNet_GED2_mean": np.mean(hpu_ged2_all), 
                "HPUNet_GED2_std": np.std(hpu_ged2_all),
                "sPUNet_IoU_mean": np.mean(spu_iou_all),
                "sPUNet_IoU_std": np.std(spu_iou_all),
                "HPUNet_IoU_mean": np.mean(hpu_iou_all),
                "HPUNet_IoU_std": np.std(hpu_iou_all),
            },
            {
                "Dataset": f"{label}_NonEmptyOnly", 
                "Evaluation_Type": "Non-Empty GT Only",
                "sPUNet_GED2_mean": np.mean(spu_ged2_nonempty),
                "sPUNet_GED2_std": np.std(spu_ged2_nonempty),
                "HPUNet_GED2_mean": np.mean(hpu_ged2_nonempty), 
                "HPUNet_GED2_std": np.std(hpu_ged2_nonempty),
                "sPUNet_IoU_mean": np.mean(spu_iou_nonempty),
                "sPUNet_IoU_std": np.std(spu_iou_nonempty),
                "HPUNet_IoU_mean": np.mean(hpu_iou_nonempty),
                "HPUNet_IoU_std": np.std(hpu_iou_nonempty),
            }
        ])

if summary_data:
    save_metrics_csv(summary_data, "summary_comparison.csv")
    print(f"\nSummary comparison saved to: {output_dir}/summary_comparison.csv")

print("\n🎉 Comparative evaluation complete!")
print("\nThe notebook evaluated both models using:")
print("  📊 Case 1: All cases (measures overall performance including empty prediction)")
print("  🎯 Case 2: Non-empty GT only (measures lesion detection when lesions present)")
print(f"\nCheck {output_dir}/ for detailed results and side-by-side visualizations!")
