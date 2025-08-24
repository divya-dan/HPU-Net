#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from torch.utils.data import DataLoader
from hpunet.data.dataset import LIDCCropsDataset
import torch

def main():
    root = Path(__file__).resolve().parents[2]  # project root
    csv_path = root / "data/lidc_crops/train.csv"

    ds = LIDCCropsDataset(csv_path=csv_path, project_root=root, image_size=128, augment=False)
    print(f"Dataset size: {len(ds)}")

    sample = ds[0]
    x, y, pm, meta = sample["image"], sample["masks"], sample["pad_mask"], sample["meta"]
    print(f"Shapes — image: {tuple(x.shape)}, masks: {tuple(y.shape)}, pad_mask: {tuple(pm.shape)}")
    print(f"Meta — split: {meta['split']}, patient: {meta['patient']}, stem: {meta['stem']}")
    print(f"Mask channels nonzero pixels: {[int(y[i].sum().item()) for i in range(4)]}")

    # small loader test
    loader = DataLoader(ds, batch_size=8, shuffle=True, num_workers=0)
    batch = next(iter(loader))
    xb, yb, pmb = batch["image"], batch["masks"], batch["pad_mask"]
    print(f"Batch — image: {tuple(xb.shape)}, masks: {tuple(yb.shape)}, pad_mask: {tuple(pmb.shape)}")
    print("OK")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
