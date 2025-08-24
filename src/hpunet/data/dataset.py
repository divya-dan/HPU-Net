from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .transforms import RandomAffine2D


def _load_png_grayscale(path: Path) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.asarray(img, dtype=np.float32)
    if arr.max() > 1.0:
        arr /= 255.0
    return arr


def _crop_indices(h: int, w: int, out_h: int, out_w: int, rng: np.random.Generator | None, random: bool) -> tuple[int, int]:
    assert out_h <= h and out_w <= w, f"crop {out_h}x{out_w} exceeds {h}x{w}"
    if random and rng is not None:
        top = int(rng.integers(0, h - out_h + 1))
        left = int(rng.integers(0, w - out_w + 1))
    else:
        top = (h - out_h) // 2
        left = (w - out_w) // 2
    return top, left


def _apply_crop(arr: np.ndarray, top: int, left: int, out_h: int, out_w: int) -> np.ndarray:
    return arr[top: top + out_h, left: left + out_w]


class LIDCCropsDataset(Dataset):
    """
    Returns a dict with:
      image: float32 tensor [1,H,W] in [0,1]
      masks: uint8 tensor [4,H,W] (missing graders are all-zero)
      pad_mask: bool tensor [H,W] (False for pixels invalidated by augmentation)
      meta: dict with csv row info
    """
    def __init__(
        self,
        csv_path: str | Path,
        project_root: str | Path | None = None,
        image_size: int = 128,
        augment: bool = False,
        seed: int = 12345,
        transform: Optional[object] = None,
    ):
        self.csv_path = Path(csv_path)
        self.project_root = Path(project_root) if project_root else self.csv_path.parent
        self.image_size = int(image_size)
        self.augment = bool(augment)
        self.rng = np.random.default_rng(seed)

        # minimal augmenter (rotation/scale/translation + flips) if augment=True
        self.transform = transform
        if self.augment and self.transform is None:
            self.transform = RandomAffine2D()

        self.df = pd.read_csv(self.csv_path)
        required_cols = [
            "split", "patient", "stem", "img_path",
            "mask_l1", "mask_l2", "mask_l3", "mask_l4",
            "has_l1", "has_l2", "has_l3", "has_l4",
        ]
        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"{self.csv_path} missing columns: {missing}")

        def _to_abs(p: str) -> Path:
            return (self.project_root / p).resolve()

        self.img_paths: List[Path] = [ _to_abs(p) for p in self.df["img_path"].tolist() ]
        self.mask_paths: List[List[Optional[Path]]] = []
        for _, row in self.df.iterrows():
            paths = []
            for k in ["mask_l1", "mask_l2", "mask_l3", "mask_l4"]:
                p = row[k]
                paths.append(_to_abs(p) if isinstance(p, str) and len(p) > 0 else None)
            self.mask_paths.append(paths)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = self.img_paths[idx]
        grader_paths = self.mask_paths[idx]

        # Load image
        img_np = _load_png_grayscale(img_path)  # [H,W]
        H, W = img_np.shape

        # Load 4 masks (missing graders -> zeros)
        masks_np = []
        for p in grader_paths:
            if p is None or not p.exists():
                m = np.zeros_like(img_np, dtype=np.uint8)
            else:
                m = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
                m = (m > 127).astype(np.uint8)
            masks_np.append(m)
        masks_np = np.stack(masks_np, axis=0)  # [4,H,W]

        # Augment (if enabled) and get validity mask
        if self.augment and self.transform is not None:
            img_np, masks_np, valid_np = self.transform(img_np, masks_np, self.rng)
        else:
            valid_np = np.ones((H, W), dtype=bool)

        # Joint crop (same indices for image/masks/pad_mask)
        out = self.image_size
        top, left = _crop_indices(img_np.shape[0], img_np.shape[1], out, out, self.rng, self.augment)
        img_np = _apply_crop(img_np, top, left, out, out)
        masks_np = np.stack([_apply_crop(m, top, left, out, out) for m in masks_np], axis=0)
        pad_mask_np = _apply_crop(valid_np, top, left, out, out)

        # Tensors
        image: Tensor = torch.from_numpy(img_np)[None, ...]  # [1,H,W]
        masks: Tensor = torch.from_numpy(masks_np)           # [4,H,W], uint8
        pad_mask: Tensor = torch.from_numpy(pad_mask_np.astype(np.bool_))  # [H,W]

        meta = {
            "split": row["split"],
            "patient": row["patient"],
            "stem": row["stem"],
            "img_path": str(img_path),
            "mask_paths": [str(p) if p is not None else "" for p in grader_paths],
            "csv_index": int(idx),
        }
        return {
            "image": image.float(),
            "masks": masks.to(torch.uint8),
            "pad_mask": pad_mask,
            "meta": meta,
        }
