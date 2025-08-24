from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


def _ensure_mask_dims(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """
    Expand pad_mask [B,H,W] or [H,W] to match per-pixel loss dims.
    """
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(1)
    return mask


def masked_bce_with_logits(
    logits: torch.Tensor,          # [B,1,H,W] or [B,H,W]
    targets: torch.Tensor,         # same spatial shape as logits
    pad_mask: torch.Tensor,        # [B,H,W] or [H,W] boolean
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    BCEWithLogits per-pixel, averaged over *valid* (pad_mask=True) pixels only.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    mask = _ensure_mask_dims(pad_mask.to(dtype=loss.dtype, device=loss.device), loss.dim())
    loss = loss * mask
    denom = mask.sum()
    if denom <= 0:
        return loss.sum() * 0.0
    return loss.sum() / denom


def masked_topk_bce_with_logits(
    logits: torch.Tensor,          # [B,1,H,W] or [B,H,W]
    targets: torch.Tensor,         # same spatial shape as logits
    pad_mask: torch.Tensor,        # [B,H,W] or [H,W] boolean
    k_frac: float = 0.02,          # keep top 2% pixels by loss
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-pixel BCEWithLogits, mask invalid pixels, then average the top-k fraction.
    """
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none", pos_weight=pos_weight)
    mask = _ensure_mask_dims(pad_mask.to(dtype=loss.dtype, device=loss.device), loss.dim())
    loss = loss * mask

    # Flatten over all dims
    flat = loss.reshape(-1)
    flat_mask = mask.reshape(-1) > 0
    valid_vals = flat[flat_mask]
    n = valid_vals.numel()
    if n == 0:
        return flat.sum() * 0.0

    k = max(1, int(n * float(k_frac)))
    topk_vals, _ = torch.topk(valid_vals, k)
    return topk_vals.mean()
