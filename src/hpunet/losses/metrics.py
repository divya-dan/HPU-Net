from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment


def binarize_logits(logits: Tensor, thr: float = 0.5) -> Tensor:
    """
    logits: [B,1,H,W] or [1,H,W] (float), interpret as logits; threshold at sigma>thr.
    Returns uint8 {0,1} with same shape.
    """
    if logits.dim() == 3:  # [1,H,W]
        logits = logits.unsqueeze(0)
    probs = logits.sigmoid()
    return (probs > thr).to(torch.uint8)


def iou_binary(a: Tensor, b: Tensor) -> Tensor:
    """
    IoU between two binary masks (uint8 {0,1}) of shape [...,H,W].
    Special case: if both are all-zero -> IoU = 1.
    Returns a scalar tensor (mean over leading dims if present).
    """
    a = a.to(torch.bool)
    b = b.to(torch.bool)
    inter = (a & b).sum(dim=(-2, -1)).float()
    a_sum = a.sum(dim=(-2, -1)).float()
    b_sum = b.sum(dim=(-2, -1)).float()
    union = a_sum + b_sum - inter
    both_empty = (a_sum == 0) & (b_sum == 0)
    # avoid div by zero: where union==0 (both empty), set IoU=1
    iou = torch.where(union > 0, inter / union.clamp_min(1e-8), torch.ones_like(union))
    return iou.mean()


def hungarian_matched_iou(gt_set: Tensor, pred_set: Tensor) -> float:
    """
    gt_set:  [G,H,W] uint8 {0,1}  (typically G=4 graders)
    pred_set:[K,H,W] uint8 {0,1}  (K samples from the model)
    Returns mean IoU over the optimal 1:1 matching between min(G,K) pairs.
    """
    G, H, W = gt_set.shape
    K = pred_set.shape[0]
    # Build IoU matrix [G,K]
    ious = torch.zeros((G, K), dtype=torch.float32)
    for g in range(G):
        for k in range(K):
            ious[g, k] = iou_binary(gt_set[g:g+1], pred_set[k:k+1])
    # Hungarian maximization -> minimize cost = 1 - IoU
    cost = (1.0 - ious).cpu().numpy()
    row_idx, col_idx = linear_sum_assignment(cost)
    # If K != G, only min(G,K) pairs are returned
    matched_ious = ious[row_idx, col_idx]
    return float(matched_ious.mean().item())


def ged2(gt_set: Tensor, pred_set: Tensor, clip_zero: bool = True) -> float:
    """
    Generalized Energy Distance squared with d = 1 - IoU.
    Uses the unbiased U-statistic estimator (can be < 0 due to finite samples).
    If clip_zero=True, returns max(est, 0.0).
    """
    G = gt_set.shape[0]
    K = pred_set.shape[0]

    def _d(a, b) -> float:
        return float((1.0 - iou_binary(a, b)).item())

    # E_gg
    if G > 1:
        s = 0.0; n = 0
        for i in range(G):
            for j in range(G):
                if i == j: continue
                s += _d(gt_set[i:i+1], gt_set[j:j+1]); n += 1
        E_gg = s / n
    else:
        E_gg = 0.0

    # E_xx
    if K > 1:
        s = 0.0; n = 0
        for i in range(K):
            for j in range(K):
                if i == j: continue
                s += _d(pred_set[i:i+1], pred_set[j:j+1]); n += 1
        E_xx = s / n
    else:
        E_xx = 0.0

    # E_gx
    s = 0.0
    for i in range(G):
        for j in range(K):
            s += _d(gt_set[i:i+1], pred_set[j:j+1])
    E_gx = s / (G * K)

    est = E_gg + E_xx - 2.0 * E_gx
    return max(est, 0.0) if clip_zero else est

