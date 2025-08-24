#!/usr/bin/env python3
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import torch

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.spu_net import ProbUNet
from hpunet.losses.metrics import (
    binarize_logits, iou_binary, hungarian_matched_iou, ged2
)

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--split", type=str, default="test", choices=["train","val","test"])
    ap.add_argument("--num-images", type=int, default=32)
    ap.add_argument("--n-prior", type=int, default=4)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--require-lesion", action="store_true",
                    help="Skip images where all grader masks are empty.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data
    csv_path = args.project_root / "data" / "lidc_crops" / f"{args.split}.csv"
    ds = LIDCCropsDataset(csv_path=csv_path, project_root=args.project_root, image_size=128, augment=False)

    # model
    model = ProbUNet(in_ch=1, base=32, z_dim=6).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    iou_rec_vals = []
    hung_vals = []
    ged_vals = []

    kept = 0
    idx = args.start_index
    while kept < args.num_images and idx < len(ds):
        sample = ds[idx]
        x = sample["image"].unsqueeze(0).to(device)   # [1,1,H,W]
        y = sample["masks"].to(device)               # [4,H,W]
        gt_set = (y > 0).to(torch.uint8)             # [4,H,W]
        if args.require_lesion and gt_set.sum().item() == 0:
            idx += 1
            continue

        # 1) Posterior reconstruction IoU: for each grader l1..l4
        per_grader = []
        for g in range(4):
            y_t = y[g:g+1].unsqueeze(0).float()      # [1,1,H,W]
            logits, _ = model(x, y_target=y_t, sample_posterior=True)
            pred = binarize_logits(logits[0])
            gt   = (y[g:g+1] > 0).to(torch.uint8)
            per_grader.append(iou_binary(pred, gt).item())
        iou_rec_vals.append(sum(per_grader) / len(per_grader))

        # 2) Prior samples set
        preds = []
        for _ in range(args.n_prior):
            logits, _ = model(x, y_target=None, sample_posterior=False)
            preds.append(binarize_logits(logits[0]).squeeze(0))  # [H,W]
        pred_set = torch.stack(preds, dim=0)  # [K,H,W]
        # gt_set = (y > 0).to(torch.uint8)      # [4,H,W]

        # Hungarian IoU
        hung_vals.append(hungarian_matched_iou(gt_set, pred_set))

        # GED^2 with d = 1 - IoU
        ged_vals.append(ged2(gt_set, pred_set))
        kept += 1
        idx += 1

    print(f"[{args.split}] N={len(iou_rec_vals)} | IoU_rec={sum(iou_rec_vals)/len(iou_rec_vals):.4f} "
           f"| Hung.IoU={sum(hung_vals)/len(hung_vals):.4f} | GED^2={sum(ged_vals)/len(ged_vals):.4f}")

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
