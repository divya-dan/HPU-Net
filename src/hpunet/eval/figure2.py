from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt


def _overlay_mask_gray(ax, img: np.ndarray, mask: np.ndarray, title: str, alpha: float = 0.45):
    """Gray CT + white fill only where mask==True (no tint elsewhere)."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    # alpha can be an array: make it nonzero only on mask pixels
    a = (mask.astype(float) * alpha)
    ax.imshow(np.ones_like(img), cmap="gray", vmin=0.0, vmax=1.0, alpha=a)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _overlay_outline(ax, img: np.ndarray, mask: np.ndarray, title: str):
    """Gray CT + white contour on the mask boundary."""
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.contour(mask.astype(float), levels=[0.5], colors="white", linewidths=1.5)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _to_np_img(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0, 1)
    return x[0].numpy()

def _to_np_mask(m: torch.Tensor, thr: float = 0.5) -> np.ndarray:
    m = m.detach().cpu()
    if m.dtype.is_floating_point:
        m = (m.sigmoid() > thr).to(torch.uint8)
    else:
        m = (m > 0).to(torch.uint8)
    return m[0].numpy().astype(bool)

def _to_np_prob(l: torch.Tensor) -> np.ndarray:
    """Sigmoid(logits) -> float [0,1], shape HxW."""
    p = l.detach().cpu().sigmoid()
    return p[0].numpy()

def _overlay_mask(ax, img: np.ndarray, mask: np.ndarray, title: str):
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    if mask.any():
        ax.imshow(mask, alpha=0.35)  # boolean overlay
    ax.set_title(title, fontsize=9)
    ax.axis("off")

def _overlay_prob(ax, img: np.ndarray, prob: np.ndarray, title: str):
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.imshow(prob, alpha=0.45, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.axis("off")

@torch.no_grad()
def make_panel(
    model,
    image,
    grader_masks,
    n_prior: int = 6,
    device: torch.device | str = "cpu",
    save_path: str | None = None,
    thr: float = 0.5,
    pad_mask: torch.Tensor | None = None,
    mode: str = "binary",         # "binary" or "prob"
    style: str = "fill-gray",     # NEW: "fill-gray" or "outline" (used when mode="binary")
) -> str:
    """
    Row 1: posterior reconstructions (one per grader).
    Row 2: prior samples (n_prior).
    mode="binary": thresholded masks; mode="prob": probability heatmaps.
    """
    model.eval()
    x = image.to(device)
    gts = grader_masks.to(device)

    # Forward passes
    post_logits, prior_logits = [], []
    for i in range(4):
        y_t = gts[:, i:i+1, :, :].float()
        logits, _ = model(x, y_target=y_t, sample_posterior=True)
        post_logits.append(logits)
    for _ in range(n_prior):
        logits, _ = model(x, y_target=None, sample_posterior=False)
        prior_logits.append(logits)

    img_np = _to_np_img(x[0])

    # Optional pad mask (True=valid)
    pad_np = None
    if pad_mask is not None:
        pm = pad_mask
        if pm.dim() == 3:
            pm = pm[0]
        pad_np = pm.detach().cpu().bool().numpy()

    def apply_pad_bool(mask_np: np.ndarray) -> np.ndarray:
        return (mask_np & pad_np) if pad_np is not None else mask_np

    def apply_pad_prob(prob_np: np.ndarray) -> np.ndarray:
        if pad_np is None:
            return prob_np
        out = prob_np.copy()
        out[~pad_np] = 0.0
        return out

    if mode == "prob":
        post_maps = [apply_pad_prob(_to_np_prob(l[0])) for l in post_logits]
        prior_maps = [apply_pad_prob(_to_np_prob(l[0])) for l in prior_logits]
    else:  # binary
        post_maps = [apply_pad_bool(_to_np_mask(l[0], thr=thr)) for l in post_logits]
        prior_maps = [apply_pad_bool(_to_np_mask(l[0], thr=thr)) for l in prior_logits]

    ncols = max(4, n_prior)
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, ncols, figsize=(2.2*ncols, 4.4))

    def draw(ax, m, title):
        if mode == "prob":
            # keep probability heatmap (use if you still want it sometimes)
            ax.imshow(img_np, cmap="gray", vmin=0.0, vmax=1.0)
            ax.imshow(m, alpha=0.45, cmap="magma", vmin=0.0, vmax=1.0)
            ax.set_title(title, fontsize=9); ax.axis("off")
        else:
            if style == "outline":
                _overlay_outline(ax, img_np, m, title)
            else:  # "fill-gray"
                _overlay_mask_gray(ax, img_np, m, title, alpha=0.45)

    # row 1: posterior per grader
    for c in range(ncols):
        ax = axes[0, c]
        if c < 4:
            title = "Post. recon l{}".format(c+1) if mode=="binary" else "Post. prob l{}".format(c+1)
            draw(ax, post_maps[c], title)
        else:
            ax.axis("off")

    # row 2: prior samples
    for c in range(ncols):
        title = "Prior sample {}".format(c+1) if mode=="binary" else "Prior prob {}".format(c+1)
        draw(axes[1, c], prior_maps[c], title)

    fig.tight_layout()
    save_path = save_path or "figure2_panel.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path