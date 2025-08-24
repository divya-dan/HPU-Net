#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from torch.utils.data import DataLoader

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.train.step_utils import reconstruction_loss_from_logits


def main():
    root = Path(__file__).resolve().parents[2]
    csv_path = root / "data/lidc_crops/train.csv"

    ds = LIDCCropsDataset(csv_path=csv_path, project_root=root, image_size=128, augment=True, seed=7)
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))

    x = batch["image"]          # [B,1,H,W]
    y = batch["masks"]          # [B,4,H,W]
    pm = batch["pad_mask"]      # [B,H,W]

    # Dummy logits (as if from a model)
    logits = torch.randn(x.size(0), 1, x.size(2), x.size(3))

    # 1) Random grader + masked BCE
    loss_bce, info_bce = reconstruction_loss_from_logits(
        logits=logits, masks=y, pad_mask=pm, strategy="random", use_topk=False
    )
    print(f"Random grader | masked BCE: {loss_bce.item():.6f} | info: {info_bce}")

    # 2) Random grader + masked top-k (2%)
    loss_topk, info_topk = reconstruction_loss_from_logits(
        logits=logits, masks=y, pad_mask=pm, strategy="random", use_topk=True, k_frac=0.02
    )
    print(f"Random grader | masked top-k BCE (2%): {loss_topk.item():.6f} | info: {info_topk}")

    # 3) Mean target (debug mode)
    loss_mean, info_mean = reconstruction_loss_from_logits(
        logits=logits, masks=y, pad_mask=pm, strategy="mean", use_topk=False
    )
    print(f"Mean target   | masked BCE: {loss_mean.item():.6f} | info: {info_mean}")

    print("OK")


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
