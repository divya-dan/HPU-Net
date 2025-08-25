#!/usr/bin/env python3
"""
sPUNet Evaluation Script
Generates Figure 2-style visualizations showing:
1. Original CT scan
2. 4 grader annotations  
3. Posterior reconstructions (conditioned on each grader)
4. Prior samples (unconditional sampling)

Same visualization format as HPUNet evaluation for direct comparison.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import sPUNet  # Use sPUNet instead of HPUNet
from hpunet.utils.config import load_config


def load_model(ckpt_path: Path, device: torch.device) -> sPUNet:
    """Load trained sPU-Net model from checkpoint"""
    ckpt = torch.load(ckpt_path, map_location=device)
    
    # Extract model config from checkpoint
    cfg = ckpt.get('cfg', {})
    model_cfg = cfg.get('model', {})
    
    # Initialize sPUNet with same config as training
    model = sPUNet(
        in_ch=1,
        base=model_cfg.get('base', 32),      # sPUNet uses base=32
        z_dim=model_cfg.get('z_dim', 6)      # sPUNet uses 6 global latents
    ).to(device)
    
    # Load weights
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"Loaded sPUNet model from step {ckpt.get('step', 'unknown')}")
    return model


def generate_posterior_reconstructions(
    model: sPUNet, 
    image: torch.Tensor, 
    grader_masks: torch.Tensor,
    device: torch.device
) -> List[torch.Tensor]:
    """
    Generate posterior reconstructions by conditioning on each grader.
    
    Args:
        model: Trained sPU-Net
        image: Input CT scan [1,1,H,W] 
        grader_masks: All grader annotations [1,4,H,W]
        
    Returns:
        List of 4 reconstruction tensors [1,1,H,W]
    """
    reconstructions = []
    
    with torch.no_grad():
        for grader_idx in range(4):
            # Get this grader's annotation [1,1,H,W]
            grader_mask = grader_masks[:, grader_idx:grader_idx+1, :, :].float()
            
            # Generate posterior reconstruction
            logits, _ = model(
                x=image,
                y_target=grader_mask,
                sample_posterior=True
            )
            
            reconstructions.append(logits)
    
    return reconstructions


def generate_prior_samples(
    model: sPUNet,
    image: torch.Tensor, 
    num_samples: int = 24,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Generate prior samples (unconditional sampling).
    
    Args:
        model: Trained sPU-Net
        image: Input CT scan [1,1,H,W]
        num_samples: Number of samples to generate
        
    Returns:
        List of sample tensors [1,1,H,W]
    """
    samples = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Sample from prior (no conditioning)
            logits, _ = model(
                x=image,
                y_target=None,
                sample_posterior=False  # Sample from prior
            )
            samples.append(logits)
    
    return samples


def create_visualization_grid_style1(
    ct_scan: np.ndarray,
    grader_masks: np.ndarray, 
    reconstructions: List[np.ndarray],
    samples: List[np.ndarray],
    save_path: Path,
    model_name: str = "sPU-Net"
):
    """
    Create Image 1 style visualization (large grid with 24 samples).
    
    Args:
        ct_scan: Original CT scan [H,W]
        grader_masks: 4 grader annotations [4,H,W] 
        reconstructions: 4 reconstructions [4,H,W]
        samples: 24 samples [24,H,W] 
        save_path: Where to save the image
        model_name: Model name for title
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f'{model_name} Evaluation Results', fontsize=16, y=0.98)
    
    # Create grid: 6 rows x 8 columns
    # Row 1: CT scan + 4 graders + 3 empty
    # Row 2: 4 reconstructions + 4 empty  
    # Rows 3-6: 24 samples (8 per row)
    
    def add_subplot_with_title(idx, img, title, cmap='gray'):
        ax = fig.add_subplot(6, 8, idx)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Row 1: CT scan and graders
    add_subplot_with_title(1, ct_scan, "CT scan")
    for i in range(4):
        add_subplot_with_title(i+2, grader_masks[i], f"grader {i+1}")
    
    # Row 2: Reconstructions  
    for i in range(4):
        add_subplot_with_title(9+i, reconstructions[i], f"recon {i+1}")
    
    # Rows 3-6: Samples (24 total, 8 per row)
    for i in range(24):
        row = 2 + (i // 8)  # Start from row 3 (index 2)
        col = (i % 8) + 1   # Columns 1-8
        subplot_idx = row * 8 + col
        add_subplot_with_title(subplot_idx, samples[i], f"s{i+1}")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to: {save_path}")


def create_visualization_grid_style2(
    ct_scan: np.ndarray,
    grader_masks: np.ndarray,
    reconstructions: List[np.ndarray], 
    samples: List[np.ndarray],
    save_path: Path,
    model_name: str = "sPU-Net"
):
    """
    Create Image 2 style visualization (compact layout with 6 samples).
    
    Args:
        ct_scan: Original CT scan [H,W]
        grader_masks: 4 grader annotations [4,H,W]
        reconstructions: 4 reconstructions [4,H,W]
        samples: First 6 samples [6,H,W]
        save_path: Where to save the image
        model_name: Model name for title
    """
    fig = plt.figure(figsize=(15, 8))
    fig.suptitle(f'{model_name} Evaluation Results', fontsize=14, y=0.95)
    
    def add_subplot_with_title(row, col, total_cols, img, title, cmap='gray'):
        ax = fig.add_subplot(3, total_cols, row * total_cols + col)
        ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Row 1: CT scan + 4 graders (5 total)
    add_subplot_with_title(0, 0, 6, ct_scan, "CT scan")
    for i in range(4):
        add_subplot_with_title(0, i+1, 6, grader_masks[i], f"grader {i+1}")
    
    # Row 2: "i) Reconstructions" - 4 reconstructions 
    for i in range(4):
        title = f"{i+1}" if i > 0 else "i) Reconstructions\n1"
        add_subplot_with_title(1, i, 6, reconstructions[i], title)
    
    # Row 3: "ii) Samples" - 6 samples
    for i in range(6):
        title = f"{i+1}" if i > 0 else "ii) Samples\n1" 
        add_subplot_with_title(2, i, 6, samples[i], title)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight') 
    plt.close()
    print(f"Saved visualization to: {save_path}")


def tensor_to_numpy(tensor: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    """Convert tensor to numpy array and binarize if needed"""
    if tensor.dim() == 4:  # [B,C,H,W]
        tensor = tensor.squeeze(0).squeeze(0)  # Remove batch and channel dims
    elif tensor.dim() == 3:  # [C,H,W] 
        tensor = tensor.squeeze(0)  # Remove channel dim
    
    # Convert to numpy
    arr = tensor.cpu().numpy()
    
    # Binarize if it looks like probabilities/logits
    if arr.max() > 1.0 or arr.min() < 0.0:
        # Assume logits, apply sigmoid then threshold
        arr = 1.0 / (1.0 + np.exp(-arr))  # Sigmoid
    
    # Apply threshold
    arr = (arr > threshold).astype(np.float32)
    
    return arr


def main():
    parser = argparse.ArgumentParser(description="sPUNet Evaluation and Visualization")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint path")
    parser.add_argument("--config", type=Path, required=True, help="Training config file")
    parser.add_argument("--data-root", type=Path, required=True, help="Data root directory")
    parser.add_argument("--output-dir", type=Path, default=Path("spu_evaluation_results"), help="Output directory")
    parser.add_argument("--num-examples", type=int, default=5, help="Number of test examples to visualize")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of prior samples per example")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and model
    cfg = load_config(args.config)
    model = load_model(args.checkpoint, device)
    
    # Load test dataset
    test_csv = args.data_root / "test.csv"
    test_ds = LIDCCropsDataset(
        csv_path=test_csv,
        project_root=args.data_root.parent,
        image_size=128,
        augment=False,  # No augmentation for evaluation
        seed=42
    )
    
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    
    print(f"Loaded {len(test_ds)} test examples")
    print(f"Generating sPU-Net visualizations for {args.num_examples} examples...")
    
    # Process examples
    for example_idx, batch in enumerate(test_loader):
        if example_idx >= args.num_examples:
            break
            
        print(f"\nProcessing example {example_idx + 1}/{args.num_examples}")
        
        # Extract data
        image = batch["image"].to(device)  # [1,1,H,W]
        masks = batch["masks"].to(device)  # [1,4,H,W] 
        
        # Generate reconstructions and samples
        print("  Generating posterior reconstructions...")
        reconstructions = generate_posterior_reconstructions(model, image, masks, device)
        
        print(f"  Generating {args.num_samples} prior samples...")
        samples = generate_prior_samples(model, image, args.num_samples, device)
        
        # Convert to numpy for visualization
        ct_scan = tensor_to_numpy(image)
        grader_masks_np = [tensor_to_numpy(masks[:, i:i+1, :, :]) for i in range(4)]
        reconstructions_np = [tensor_to_numpy(recon) for recon in reconstructions] 
        samples_np = [tensor_to_numpy(sample) for sample in samples]
        
        # Create both visualization styles
        print("  Creating visualizations...")
        
        # Style 1: Large grid (like Image 1)
        save_path_1 = args.output_dir / f"spu_example_{example_idx+1}_style1.png"
        create_visualization_grid_style1(
            ct_scan, np.array(grader_masks_np), reconstructions_np, samples_np, 
            save_path_1, model_name="sPU-Net"
        )
        
        # Style 2: Compact layout (like Image 2) 
        save_path_2 = args.output_dir / f"spu_example_{example_idx+1}_style2.png"
        create_visualization_grid_style2(
            ct_scan, np.array(grader_masks_np), reconstructions_np, samples_np[:6], 
            save_path_2, model_name="sPU-Net"
        )
    
    print(f"\nsPU-Net evaluation complete! Results saved to: {args.output_dir}")
    print("\nTo compare with HPU-Net results:")
    print("1. Run HPU-Net evaluation: python evaluate_hpu.py ...")
    print("2. Run sPU-Net evaluation: python evaluate_spu.py ...")
    print("3. Compare output images side by side")


if __name__ == "__main__":
    main()
