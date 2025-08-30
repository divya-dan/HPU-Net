#!/usr/bin/env python3
"""
Build CSV manifests for LIDC crops.

For each split (train/val/test), this script scans:
  data_root/<split>/
    images/<patient_id>/*.png
    gt/<patient_id>/*_l[0..3].png

It writes one CSV per split with columns:
  split,patient,stem,img_path,
  mask_l0,mask_l1,mask_l2,mask_l3,
  has_l0,has_l1,has_l2,has_l3,n_masks

Paths are written relative to the project root (useful for portability).
If a grader's mask is missing or completely black, the path is empty and has_l#=False (your Dataset
will substitute an all-zeros mask for such channels).
"""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

def is_mask_empty(mask_path: Path) -> bool:
    """Check if mask image is completely black (all pixels are 0)."""
    try:
        img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return True
        return np.all(img == 0)
    except Exception:
        return True

def collect_split(split_dir: Path, split_name: str, project_root: Path) -> pd.DataFrame:
    images_dir = split_dir / "images"
    gt_dir = split_dir / "gt"

    if not images_dir.is_dir() or not gt_dir.is_dir():
        raise FileNotFoundError(f"Expected {images_dir} and {gt_dir} to exist.")

    rows = []
    # iterate patients deterministically
    for patient_dir in sorted(images_dir.glob("*")):
        if not patient_dir.is_dir():
            continue
        patient = patient_dir.name
        # iterate images deterministically
        for img_path in sorted(patient_dir.glob("*.png")):
            stem = img_path.stem  # e.g., "z-<ZPOS>_c<CROP>"
            # Masks for graders 0..3
            mask_paths = []
            has_flags = []
            for l in range(0, 4):  # Changed from range(1, 5) to range(0, 4)
                mpath = gt_dir / patient / f"{stem}_l{l}.png"
                if mpath.exists():
                    # File exists, include path regardless of content
                    mask_paths.append(mpath)
                    # But set has_flag based on whether mask is empty
                    has_flags.append(not is_mask_empty(mpath))
                else:
                    # File doesn't exist
                    mask_paths.append(None)
                    has_flags.append(False)

            rel_img = img_path.resolve().relative_to(project_root.resolve())
            rel_masks = [
                (p.resolve().relative_to(project_root.resolve())).as_posix() if p else ""
                for p in mask_paths
            ]
            row = {
                "split": split_name,
                "patient": patient,
                "stem": stem,
                "img_path": rel_img.as_posix(),
                "mask_l0": rel_masks[0],  # Changed from mask_l1 to mask_l0
                "mask_l1": rel_masks[1],
                "mask_l2": rel_masks[2],
                "mask_l3": rel_masks[3],  # Changed from mask_l4 to mask_l3
                "has_l0": has_flags[0],   # Changed from has_l1 to has_l0
                "has_l1": has_flags[1],
                "has_l2": has_flags[2],
                "has_l3": has_flags[3],   # Changed from has_l4 to has_l3
                "n_masks": int(sum(has_flags)),
            }
            rows.append(row)

    df = pd.DataFrame(rows).sort_values(["patient", "stem"]).reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Path to lidc_crops directory (contains train/val/test).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Where to write CSVs (typically the same data-root).")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                    help="Which splits to process.")
    ap.add_argument("--project-root", type=Path, default=Path.cwd(),
                    help="Base to which CSV paths are made relative (default: CWD).")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    for split in args.splits:
        split_dir = args.data_root / split
        df = collect_split(split_dir, split, args.project_root)
        out_csv = args.out / f"{split}.csv"
        df.to_csv(out_csv, index=False)
        # brief summary
        counts = df["n_masks"].value_counts().sort_index()
        total_imgs = len(df)
        print(f"[{split}] images={total_imgs}  mask-count distribution:", end=" ")
        print(", ".join(f"{k}→{counts.get(k,0)}" for k in range(5)))
        print(f"CSV written: {out_csv}")

if __name__ == "__main__":
    main()
