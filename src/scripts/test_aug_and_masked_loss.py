#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.losses.topk_ce import masked_bce_with_logits, masked_topk_bce_with_logits


def main():
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data/lidc_crops/train.csv"

    ds = LIDCCropsDataset(
        csv_path=csv_path,
        project_root=root,
        image_size=128,
        augment=True,  # enable our minimal affine augmenter
        seed=2025,
    )

    sample = ds[0]
    x, y, pm = sample["image"], sample["masks"], sample["pad_mask"]
    print(f"Sample shapes: x {tuple(x.shape)}, y {tuple(y.shape)}, pad {tuple(pm.shape)}")
    print(f"Pad mask valid ratio: {pm.float().mean().item():.4f}")

    # Fake logits to test the masked losses
    logits = torch.randn(1, 1, 128, 128)
    target = y[0:1, None, ...].float()  # use grader-1 as target shape [1,1,128,128]
    pm_b = pm[None, ...]                # [1,128,128]

    m_bce = masked_bce_with_logits(logits, target, pm_b)
    m_topk = masked_topk_bce_with_logits(logits, target, pm_b, k_frac=0.02)
    print(f"masked BCE: {m_bce.item():.6f} | masked top-k BCE (2%): {m_topk.item():.6f}")

    # Dataloader quick check with augment=True
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    print(f"Batch image: {tuple(batch['image'].shape)}, masks: {tuple(batch['masks'].shape)}, pad: {tuple(batch['pad_mask'].shape)}")
    print("OK")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
