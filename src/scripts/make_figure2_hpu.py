#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import torch

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.hpu_net import HPUNet
from hpunet.eval.figure2 import make_panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--n-prior", type=int, default=6)
    ap.add_argument("--out", type=Path, default=Path("runs/fig2_hpu_panel.png"))
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--mode", type=str, default="binary", choices=["binary","prob"])
    ap.add_argument("--style", type=str, default="fill-gray", choices=["fill-gray","outline"])  # NEW
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    csv_path = args.project_root / "data" / "lidc_crops" / f"{args.split}.csv"
    ds = LIDCCropsDataset(csv_path=csv_path, project_root=args.project_root, image_size=128, augment=False)
    samp = ds[args.index]
    x = samp["image"].unsqueeze(0).to(device)     # [1,1,H,W]
    y = samp["masks"].unsqueeze(0).to(device)     # [1,4,H,W]
    pm = samp["pad_mask"].unsqueeze(0)            # [1,H,W], keep on CPU for plotting

    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)

    pm = samp["pad_mask"].unsqueeze(0)
    out_path = make_panel(
        model, x, y, n_prior=args.n_prior, device=device,
        save_path=str(args.out), thr=args.thr, pad_mask=pm,
        mode=args.mode, style=args.style   # pass it through
    )
    print(f"Saved panel to: {out_path}")
    meta = samp["meta"]
    print(f"Sample info — split: {meta['split']}, patient: {meta['patient']}, stem: {meta['stem']}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
