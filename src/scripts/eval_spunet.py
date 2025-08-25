import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import ProbUNet
from hpunet.metrics import hungarian_matched_iou, ged2, binarize_logits


def visualize_samples_grid(images, gt_masks, pred_samples, save_path=None, 
                          num_display=8, num_samples=16):
    """
    Create Figure 10-style visualization grid.
    
    Args:
        images: [B,1,H,W] input CT images
        gt_masks: [B,4,H,W] ground truth masks from 4 graders  
        pred_samples: [B,num_samples,1,H,W] model predictions
        save_path: where to save the figure
        num_display: how many cases to show horizontally
        num_samples: how many samples per case to show
    """
    B = min(images.size(0), num_display)
    
    # Create grid: rows = [image, graders(4), samples(num_samples)]
    num_rows = 1 + 4 + num_samples  
    fig = plt.figure(figsize=(B * 2, num_rows * 1.5))
    gs = gridspec.GridSpec(num_rows, B, wspace=0.02, hspace=0.02)
    
    for col in range(B):
        # Input image (grayscale CT scan)
        ax = fig.add_subplot(gs[0, col])
        img = images[col, 0].cpu().numpy()
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel('Input', rotation=0, ha='right', va='center')
            
        # Ground truth masks (4 graders)
        for grader in range(4):
            ax = fig.add_subplot(gs[1 + grader, col])
            mask = gt_masks[col, grader].cpu().numpy()
            ax.imshow(mask, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'Grader {grader+1}', rotation=0, ha='right', va='center')
                
        # Model samples 
        for sample_idx in range(num_samples):
            ax = fig.add_subplot(gs[5 + sample_idx, col])
            # Binarize logits to get binary masks
            sample_logits = pred_samples[col, sample_idx:sample_idx+1]  # [1,1,H,W]
            sample_mask = binarize_logits(sample_logits)[0, 0].cpu().numpy()
            ax.imshow(sample_mask, cmap='gray', vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f'Sample {sample_idx+1}', rotation=0, ha='right', va='center')
    
    plt.suptitle('Probabilistic U-Net Results on LIDC Dataset', fontsize=14)
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    plt.close()


def evaluate_model(model_path: Path, data_root: Path, project_root: Path, 
                   save_dir: Path, num_samples: int = 16, num_cases: int = 15):
    """
    Evaluate trained model and create visualizations + metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Load test data
    test_csv = data_root / "test.csv"  # or "val.csv"
    test_ds = LIDCCropsDataset(
        csv_path=test_csv, project_root=project_root,
        image_size=128, augment=False  # No augmentation for evaluation
    )
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=2)
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics storage
    all_hungarian_ious = []
    all_geds = []
    
    print("Generating samples and computing metrics...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx >= num_cases // 8:  # Limit number of cases
                break
                
            x = batch["image"].to(device)          # [B,1,H,W]
            gt_masks = batch["masks"].to(device)   # [B,4,H,W] 
            pad_mask = batch["pad_mask"].to(device)  # [B,H,W]
            
            # Generate multiple samples from the model
            pred_samples = model.sample_multiple(x, num_samples=num_samples)
            # pred_samples: [B, num_samples, 1, H, W]
            
            # Compute metrics for each case in batch
            for b in range(x.size(0)):
                # Get ground truth set (4 graders) and prediction set (num_samples)
                gt_set = gt_masks[b]  # [4,H,W]
                pred_set = binarize_logits(pred_samples[b])  # [num_samples,1,H,W] -> [num_samples,H,W]
                pred_set = pred_set.squeeze(1)  # [num_samples,H,W]
                
                # Hungarian-matched IoU
                hiou = hungarian_matched_iou(gt_set, pred_set)
                all_hungarian_ious.append(hiou)
                
                # Generalized Energy Distance
                ged = ged2(gt_set, pred_set)
                all_geds.append(ged)
            
            # Create visualization for this batch
            viz_path = save_dir / f"batch_{batch_idx:03d}_samples.png"
            visualize_samples_grid(
                images=x,
                gt_masks=gt_masks,
                pred_samples=pred_samples,
                save_path=viz_path,
                num_display=min(8, x.size(0)),
                num_samples=min(16, num_samples)
            )
    
    # Print metrics summary
    mean_hiou = np.mean(all_hungarian_ious)
    mean_ged = np.mean(all_geds)
    print(f"\nEvaluation Results:")
    print(f"Hungarian-matched IoU: {mean_hiou:.4f} ± {np.std(all_hungarian_ious):.4f}")
    print(f"Generalized Energy Distance: {mean_ged:.4f} ± {np.std(all_geds):.4f}")
    
    # Save metrics
    metrics_path = save_dir / "metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"Hungarian-matched IoU: {mean_hiou:.4f} ± {np.std(all_hungarian_ious):.4f}\n")
        f.write(f"Generalized Energy Distance: {mean_ged:.4f} ± {np.std(all_geds):.4f}\n")
        f.write(f"Number of cases evaluated: {len(all_hungarian_ious)}\n")
    
    return mean_hiou, mean_ged


if __name__ == "__main__":
    import argparse
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True, help="Path to trained model checkpoint")
    ap.add_argument("--data-root", type=Path, required=True, help="Path to LIDC data directory")
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--save-dir", type=Path, default=Path("evaluation_results"))
    ap.add_argument("--num-samples", type=int, default=16)
    ap.add_argument("--num-cases", type=int, default=15)
    
    args = ap.parse_args()
    
    evaluate_model(
        model_path=args.model_path,
        data_root=args.data_root, 
        project_root=args.project_root,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        num_cases=args.num_cases
    )