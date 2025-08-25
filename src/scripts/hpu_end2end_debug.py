#!/usr/bin/env python3
"""
End-to-end HPUNet debug:
- train for N steps (GECO + optional top-k BCE + pos_weight balancing)
- evaluate on a subset (IoU_rec, Hungarian IoU, GED^2)
- render a black&white grid like the paper (CT | graders | reconstructions | samples)

Example:
  export PYTHONPATH="$(pwd)/src"
  python src/scripts/hpu_end2end_debug.py \
      --project-root "$(pwd)" \
      --data-root "data/lidc_crops" \
      --max-steps 2000 \
      --batch-size 8 \
      --n-cols 8 \
      --n-sample-rows 3 \
      --outdir "runs/hpu_debug"
"""
from __future__ import annotations
import sys
from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ensure src/ on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from hpunet.data.dataset import LIDCCropsDataset
from hpunet.models.hpu_net import HPUNet
from hpunet.losses.topk_ce import masked_bce_with_logits, masked_topk_bce_with_logits
from hpunet.train.step_utils import select_targets_from_graders
from hpunet.losses.metrics import binarize_logits, iou_binary, hungarian_matched_iou, ged2

from hpunet.utils.logger import Logger


# --------------------- utils ---------------------

def set_seed(seed: int = 123):
    import numpy as _np
    random.seed(seed); _np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def auto_pos_weight(y_target: torch.Tensor, pad_mask: torch.Tensor, clip: float = 20.0) -> torch.Tensor | None:
    """Per-batch positive class weight for BCE: (negatives / positives)."""
    valid = pad_mask.float().unsqueeze(1)  # [B,1,H,W]
    n_pos = (y_target * valid).sum()
    n_valid = valid.sum()
    if n_pos > 0:
        w = ((n_valid - n_pos) / (n_pos + 1e-6)).clamp(1.0, clip)
        return torch.tensor([float(w.item())], device=y_target.device)
    return None

@torch.no_grad()
def pick_lesion_indices(ds: LIDCCropsDataset, n: int, start: int = 0) -> list[int]:
    out, i = [], start
    while len(out) < n and i < len(ds):
        if (ds[i]["masks"] > 0).sum().item() > 0:
            out.append(i)
        i += 1
    while len(out) < n and len(out) > 0:
        out.append(out[-1])
    return out

# simple inline GECO (moving-average constraint)
class SimpleGECO:
    def __init__(self, kappa: float = 0.60, alpha: float = 0.99, lambda_init: float = 1.0, step_size: float = 0.05):
        self.kappa = float(kappa)
        self.alpha = float(alpha)
        self.log_lambda = float(np.log(lambda_init))
        self.step_size = float(step_size)
        self.C_bar = 0.0

    def update(self, recon_scalar: float):
        C = float(recon_scalar) - self.kappa
        self.C_bar = self.alpha * self.C_bar + (1.0 - self.alpha) * C
        self.log_lambda += self.step_size * self.C_bar
        lam = float(np.exp(self.log_lambda))
        return {"lambda": lam, "C": C, "C_bar": self.C_bar}


# --------------------- training ---------------------

def train_hpu_for_steps(
    project_root: Path, data_root: Path, outdir: Path,
    max_steps: int = 2000, batch_size: int = 8, lr: float = 1e-4,
    seed: int = 123, num_workers: int = 2,
    use_topk: bool = True, k_frac: float = 0.02,
    geco_kappa: float = 0.60, geco_alpha: float = 0.99,
    geco_lambda_init: float = 1.0, geco_step_size: float = 0.05,
    pos_weight: str | float | None = "auto", pos_weight_clip: float = 20.0,
    logger: Logger | None = None,
) -> tuple[Path, Logger]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)
    logger = logger or Logger(outdir, use_tensorboard=True)

    logger.log_scalars(0, {
        "train/L": 0.0, "train/recon": 0.0, "train/kl": 0.0,
        "train/lambda": 0.0, "train/C": 0.0, "train/C_bar": 0.0, "train/lr": lr,
        "eval/num": 0.0, "eval/IoU_rec": 0.0, "eval/HungIoU": 0.0, "eval/GED2": 0.0,
    })

    # data
    train_csv = data_root / "train.csv"
    train_ds = LIDCCropsDataset(csv_path=train_csv, project_root=project_root,
                                image_size=128, augment=True, seed=seed)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=True, drop_last=True)

    # model/opt
    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    geco = SimpleGECO(geco_kappa, geco_alpha, geco_lambda_init, geco_step_size)

    model.train()
    it = iter(loader)
    for step in range(1, max_steps + 1):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        x   = batch["image"].to(device, non_blocking=True)       # [B,1,H,W]
        y_g = batch["masks"].to(device, non_blocking=True)       # [B,4,H,W]
        pm  = batch["pad_mask"].to(device, non_blocking=True)    # [B,H,W]

        # choose a grader target for posterior path
        y_target, _ = select_targets_from_graders(y_g, strategy="random")  # [B,1,H,W]

        # forward (posterior)
        logits, info = model(x, y_target=y_target, sample_posterior=True)
        kl = info.get("kl", 0.0)
        if isinstance(kl, torch.Tensor) and kl.ndim > 0:
            kl = kl.mean()

        # class balancing
        pw = None
        if pos_weight == "auto":
            pw = auto_pos_weight(y_target, pm, clip=pos_weight_clip)
        elif isinstance(pos_weight, (int, float)):
            pw = torch.tensor([float(pos_weight)], device=device)

        # reconstruction (masked BCE or masked top-k)
        if use_topk:
            recon = masked_topk_bce_with_logits(logits, y_target, pm, k_frac=k_frac, pos_weight=pw)
        else:
            recon = masked_bce_with_logits(logits, y_target, pm, pos_weight=pw)

        # GECO update & total loss
        ge = geco.update(float(recon.item()))
        lam = ge["lambda"]
        # Loss uses *current* recon scalar for constraint; backprop through recon only.
        loss = recon + kl + lam * (recon - geco.kappa)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # logging
        logger.log_scalars(step, {
            "train/L": float(loss),
            "train/recon": float(recon),
            "train/kl": float(kl),
            "train/lambda": float(lam),
            "train/C": float(ge["C"]),
            "train/C_bar": float(ge["C_bar"]),
            "train/lr": opt.param_groups[0]["lr"],
        })
        if step % 50 == 0 or step == 1:
            print(f"[step {step:5d}] L={float(loss):.4f} recon={float(recon):.4f} kl={float(kl):.4f} "
                  f"lam={lam:.3f} C={ge['C']:.4f} C̄={ge['C_bar']:.4f}")

    ckpt = outdir / "hpu_last.pth"
    torch.save({"step": max_steps, "model": model.state_dict()}, ckpt)
    print(f"Checkpoint saved to: {ckpt}")
    return ckpt, logger


# --------------------- evaluation ---------------------
@torch.no_grad()
def evaluate_hpu(
    ckpt: Path,
    project_root: Path,
    data_root: Path,
    split: str = "test",
    num_images: int = 64,
    n_prior: int = 16,
    start_index: int = 0,
    require_lesion: bool = True,   # kept for backward-compat; lesion tiles are picked explicitly
    logger: Logger | None = None,
    step: int | None = None,
    thr: float = 0.5,              # NEW: probability threshold for binarization
):
    """
    Evaluate HPUNet on 'num_images' tiles that *contain lesions* (picked from 'start_index'),
    using a configurable binarization threshold 'thr'.
    Logs IoU_rec, Hungarian IoU, and GED^2 (clipped at 0) via the provided logger.
    """

    # ---- small helper so we only evaluate on lesion-containing tiles ----
    def _pick_lesion_indices(ds: LIDCCropsDataset, n: int, start: int = 0) -> list[int]:
        out, i = [], start
        while len(out) < n and i < len(ds):
            s = ds[i]
            if (s["masks"] > 0).sum().item() > 0:
                out.append(i)
            i += 1
        return out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LIDCCropsDataset(
        csv_path=data_root / f"{split}.csv",
        project_root=project_root,
        image_size=128,
        augment=False,
    )

    # pick explicit lesion indices (ignores require_lesion)
    idxs = _pick_lesion_indices(ds, num_images, start=start_index)
    if len(idxs) == 0:
        print(f"[{split}] No lesion tiles found from start_index={start_index}.")
        if logger is not None:
            s = 0 if step is None else int(step)
            logger.log_scalars(s, {
                "eval/num": 0.0,
                "eval/IoU_rec": 0.0,
                "eval/HungIoU": 0.0,
                "eval/GED2": 0.0,
            })
        return

    # load model
    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    vals_rec, vals_hung, vals_ged = [], [], []

    for idx in idxs:
        s = ds[idx]
        x = s["image"].unsqueeze(0).to(device)    # [1,1,H,W]
        y = s["masks"].to(device)                 # [4,H,W]
        gt_set = (y > 0).to(torch.uint8)          # boolean/uint8 set of 4 graders

        # posterior recon IoU per grader (averaged)
        ious = []
        for g in range(4):
            y_t = y[g:g+1].unsqueeze(0).float()   # [1,1,H,W]
            logits, _ = model(x, y_target=y_t, sample_posterior=True)
            pred = binarize_logits(logits[0], thr)              # <-- threshold used here
            ious.append(iou_binary(pred, gt_set[g:g+1]).item())
        vals_rec.append(sum(ious) / 4.0)

        # prior sample set
        preds = []
        for _ in range(n_prior):
            logits, _ = model(x, y_target=None, sample_posterior=False)
            preds.append(binarize_logits(logits[0], thr).squeeze(0))
        pred_set = torch.stack(preds, dim=0)      # [n_prior,H,W]

        vals_hung.append(hungarian_matched_iou(gt_set, pred_set))
        vals_ged.append(max(ged2(gt_set, pred_set), 0.0))

    def _mean(v): return sum(v) / max(len(v), 1)
    iou_rec = _mean(vals_rec)
    hung = _mean(vals_hung)
    ged = _mean(vals_ged)

    print(f"[{split}] N={len(idxs)} | IoU_rec={iou_rec:.4f} | Hung.IoU={hung:.4f} | GED^2={ged:.4f}")
    if logger is not None:
        s = 0 if step is None else int(step)
        logger.log_scalars(s, {
            "eval/num": float(len(idxs)),
            "eval/IoU_rec": float(iou_rec),
            "eval/HungIoU": float(hung),
            "eval/GED2": float(ged),
        })


# --------------------- figure (grid) ---------------------

@torch.no_grad()
def render_grid(
    ckpt: Path, project_root: Path, data_root: Path, out_png: Path,
    split: str = "test", n_cols: int = 8, n_sample_rows: int = 3, start_index: int = 0,
    dividers: bool = True, divider_color: str = "red", divider_lw: float = 2.0,
):
    """
    Grid rows:
      0: CT scans (grayscale)
      1..4: 4 expert graders (white mask on black)
      5..8: 4 posterior reconstructions (white mask)
      9.. : n_sample_rows prior samples (white mask)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = LIDCCropsDataset(csv_path=data_root / f"{split}.csv",
                          project_root=project_root, image_size=128, augment=False)
    idxs = pick_lesion_indices(ds, n_cols, start=start_index)

    model = HPUNet(in_ch=1, base=32, z_ch=8).to(device)
    ckpt_obj = torch.load(ckpt, map_location=device)
    model.load_state_dict(ckpt_obj["model"], strict=True)
    model.eval()

    imgs, graders, recons, samples = [], [], [], []
    for idx in idxs:
        s = ds[idx]
        x = s["image"].unsqueeze(0).to(device)         # [1,1,H,W]
        y = (s["masks"] > 0)                           # [4,H,W] bool
        pm = s["pad_mask"].bool().numpy()              # [H,W] bool
        imgs.append(s["image"][0].numpy())
        graders.append((y.numpy() & pm))

        # posterior recon per grader
        col_recons = []
        for g in range(4):
            y_t = y[g:g+1].float().unsqueeze(0).to(device)
            l, _ = model(x, y_target=y_t, sample_posterior=True)
            pred = (l.sigmoid()[0, 0] > 0.5).detach().cpu().numpy()
            col_recons.append(pred & pm)
        recons.append(np.stack(col_recons, 0))

        # prior samples (one per sample row)
        col_samples = []
        for _ in range(n_sample_rows):
            l, _ = model(x, y_target=None, sample_posterior=False)
            pr = (l.sigmoid()[0, 0] > 0.5).detach().cpu().numpy()
            col_samples.append(pr & pm)
        samples.append(np.stack(col_samples, 0))  # [R,H,W] for this column

    imgs    = np.stack(imgs, 0)                     # [C,H,W]
    graders = np.stack(graders, 0)                   # [C,4,H,W]
    recons  = np.stack(recons, 0)                    # [C,4,H,W]
    samples = np.stack(samples, 0)                   # [C,R,H,W]
    samples = np.transpose(samples, (1, 0, 2, 3))    # -> [R,C,H,W]

    H, W = imgs.shape[-2:]
    n_rows = 1 + 4 + 4 + n_sample_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7*n_cols, 1.7*n_rows))
    if n_cols == 1:
        axes = np.expand_dims(axes, 1)

    # Row 0: CT
    for c in range(n_cols):
        ax = axes[0, c]; ax.imshow(imgs[c], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 1..4: graders
    for r in range(4):
        for c in range(n_cols):
            ax = axes[1+r, c]; ax.imshow(graders[c, r], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 5..8: posterior recons
    for r in range(4):
        for c in range(n_cols):
            ax = axes[5+r, c]; ax.imshow(recons[c, r], cmap="gray", vmin=0, vmax=1); ax.axis("off")
    # Rows 9..: prior samples
    for r in range(n_sample_rows):
        for c in range(n_cols):
            ax = axes[9+r, c]; ax.imshow(samples[r, c], cmap="gray", vmin=0, vmax=1); ax.axis("off")

    # section labels
    fig.text(0.005, 1.00 - (0.5 / n_rows), "CT scans", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((1 + 2.0) / n_rows), "4 expert graders", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((5 + 2.0) / n_rows), "reconstructions", va="center", ha="left", rotation=90, fontsize=10)
    fig.text(0.005, 1.00 - ((9 + (n_sample_rows/2)) / n_rows), "samples", va="center", ha="left", rotation=90, fontsize=10)

    fig.tight_layout(rect=[0.03, 0.02, 1, 0.98])
    # dividers between CT | graders | recons
    if dividers:
        plt.draw()
        def _hline_between(r: int):
            if r + 1 >= n_rows: return
            left  = min(axes[i, 0].get_position().x0 for i in range(n_rows))
            right = max(axes[i, -1].get_position().x1 for i in range(n_rows))
            y_bot_this = axes[r, 0].get_position().y0
            y_top_next = axes[r+1, 0].get_position().y1
            y = 0.5 * (y_bot_this + y_top_next)
            fig.add_artist(Line2D([left, right], [y, y], transform=fig.transFigure,
                                  color=divider_color, linewidth=divider_lw))
        for r in (0, 4, 8):
            if r < n_rows - 1: _hline_between(r)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved grid figure to: {out_png}")


# --------------------- CLI ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path.cwd())
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("runs/hpu_debug"))
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--use-topk", action="store_true", help="use masked top-k BCE")
    ap.add_argument("--k-frac", type=float, default=0.02)
    # GECO:
    ap.add_argument("--geco-kappa", type=float, default=0.60)
    ap.add_argument("--geco-alpha", type=float, default=0.99)
    ap.add_argument("--geco-lambda-init", type=float, default=1.0)
    ap.add_argument("--geco-step-size", type=float, default=0.05)
    # class balance
    ap.add_argument("--pos-weight", type=str, default="auto", help='"auto", float-as-string, or "none"')
    ap.add_argument("--pos-weight-clip", type=float, default=20.0)
    # grid
    ap.add_argument("--n-cols", type=int, default=8)
    ap.add_argument("--n-sample-rows", type=int, default=3)
    ap.add_argument("--start-index", type=int, default=0)
    ap.add_argument("--thr", type=float, default=0.5, help="probability threshold for binarization")
    args = ap.parse_args()

    project_root = args.project_root
    data_root = args.data_root or (project_root / "data" / "lidc_crops")
    outdir = args.outdir

    # parse pos_weight
    pw: str | float | None
    if args.pos_weight.lower() == "none":
        pw = None
    else:
        try:
            pw = float(args.pos_weight)
        except ValueError:
            pw = "auto"

    ckpt, logger = train_hpu_for_steps(
        project_root, data_root, outdir,
        max_steps=args.max_steps, batch_size=args.batch_size, lr=1e-4,
        seed=123, num_workers=args.num_workers,
        use_topk=args.use_topk, k_frac=args.k_frac,
        geco_kappa=args.geco_kappa, geco_alpha=args.geco_alpha,
        geco_lambda_init=args.geco_lambda_init, geco_step_size=args.geco_step_size,
        pos_weight=pw, pos_weight_clip=args.pos_weight_clip,
    )

    evaluate_hpu(
        ckpt, project_root, data_root,
        split="test", num_images=64, n_prior=16,
        start_index=args.start_index, require_lesion=True,
        logger=logger, step=args.max_steps, thr=args.thr
    )

    render_grid(
        ckpt, project_root, data_root,
        out_png=outdir / "fig_grid.png",
        split="test", n_cols=args.n_cols,
        n_sample_rows=args.n_sample_rows, start_index=args.start_index,
        dividers=True, divider_color="red", divider_lw=2.0
    )
    logger.save_summary_figure(outdir / "metrics_summary.png", ema=0.95)
    logger.close()


if __name__ == "__main__":
    main()
