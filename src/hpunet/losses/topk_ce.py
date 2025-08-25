from __future__ import annotations
from typing import Optional
import torch
import torch.nn.functional as F


# returns: per-image SUMS (shape [B]) and per-image COUNTS (shape [B])
def masked_bce_sum_per_image(logits, target, pad_mask, pos_weight=None):
    # logits,target: [B,1,H,W]; pad_mask: [B,H,W] (bool/byte)
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none', pos_weight=pos_weight
    ).squeeze(1)                                   # [B,H,W]
    valid = pad_mask.bool()
    sums  = (bce * valid).flatten(1).sum(dim=1)    # [B]
    counts= valid.flatten(1).sum(dim=1)            # [B]
    return sums, counts

def masked_topk_bce_sum_per_image(logits, target, pad_mask, k_frac=0.02, pos_weight=None):
    bce = nn.functional.binary_cross_entropy_with_logits(
        logits, target, reduction='none', pos_weight=pos_weight
    ).squeeze(1)                                   # [B,H,W]
    valid = pad_mask.bool()
    B, H, W = bce.shape
    sums  = logits.new_zeros(B)
    counts= torch.zeros(B, dtype=torch.long, device=logits.device)
    for i in range(B):
        v = valid[i].view(-1)
        n = v.sum().item()
        if n == 0:
            continue
        k = max(1, int(round(k_frac * n)))
        vals = bce[i].view(-1)[v]
        topk = torch.topk(vals, k, largest=True).values
        sums[i]   = topk.sum()
        counts[i] = k
    return sums, counts


def _ensure_mask_dims(mask: torch.Tensor, target_ndim: int) -> torch.Tensor:
    """
    Expand pad_mask [B,H,W] or [H,W] to match per-pixel loss dims.
    """
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(1)
    return mask


# ========================================
# ADD THESE NEW FUNCTIONS HERE:
# ========================================

def _sample_gumbel(shape: torch.Size, device: torch.device, eps: float = 1e-20) -> torch.Tensor:
    """Sample from Gumbel(0, 1) distribution"""
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)

def masked_stochastic_topk_bce_with_logits(
    logits: torch.Tensor,          # [B,1,H,W] or [B,H,W]
    targets: torch.Tensor,         # same spatial shape as logits
    pad_mask: torch.Tensor,        # [B,H,W] or [H,W] boolean
    k_frac: float = 0.02,          # keep top 2% pixels by loss
    pos_weight: Optional[torch.Tensor] = None,
    deterministic: bool = False,   # if True, use deterministic top-k
) -> torch.Tensor:
    """
    STOCHASTIC top-k BCE using Gumbel-Softmax sampling (as in the paper).
    This samples the top-k pixels probabilistically based on their loss values.
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
    
    # Normalize loss values to create probability distribution
    norm_loss = valid_vals / (valid_vals.sum() + 1e-8)
    
    if deterministic:
        # Original deterministic version
        scores = torch.log(norm_loss + 1e-8)
    else:
        # STOCHASTIC: Add Gumbel noise for sampling (as in paper)
        gumbel_noise = _sample_gumbel(norm_loss.shape, norm_loss.device)
        scores = torch.log(norm_loss + 1e-8) + gumbel_noise
    
    # Select top-k based on scores
    topk_vals, topk_indices = torch.topk(scores, k)
    selected_losses = valid_vals[topk_indices]
    
    return selected_losses.mean()

# ========================================
# EXISTING FUNCTIONS (keep as-is):
# ========================================

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

def compute_pos_weight_from_batch(targets: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    # targets: [B,1,H,W] (0/1), pad_mask: [B,H,W]
    m = pad_mask.bool()
    pos = targets.squeeze(1)[m].sum().clamp(min=1)  # avoid div by zero
    neg = m.sum() - pos
    # BCEWithLogits expects ratio of negatives to positives
    return (neg.float() / pos.float()).detach()


# ========================================
# UPDATE THIS FUNCTION:
# ========================================

def make_recon_loss(
        logits: torch.Tensor,
        y_target: torch.Tensor,
        pad_mask: torch.Tensor,
        use_topk: bool,
        k_frac: float = 0.02,
        stochastic_topk: bool = True,  # NEW: use stochastic top-k as in paper
    ) -> torch.Tensor:
    pos_w = compute_pos_weight_from_batch(y_target, pad_mask)
    if use_topk:
        return masked_stochastic_topk_bce_with_logits(  # CHANGED: use stochastic version
            logits, y_target, pad_mask, 
            k_frac=k_frac, 
            pos_weight=pos_w,
            deterministic=not stochastic_topk
        )
    else:
        return masked_bce_with_logits(logits, y_target, pad_mask, pos_weight=pos_w)