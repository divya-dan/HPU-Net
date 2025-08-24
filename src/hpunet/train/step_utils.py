from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from torch import Tensor

from hpunet.losses.topk_ce import (
    masked_bce_with_logits,
    masked_topk_bce_with_logits,
)


def select_targets_from_graders(
    masks: Tensor,                 # [B,4,H,W] uint8 {0,1}
    strategy: str = "random",      # "random" | "fixed" | "mean"
    fixed_index: Optional[int] = None,  # used if strategy == "fixed", in {0,1,2,3}
    rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Returns targets shaped [B,1,H,W] in float32 ∈ {0,1} (or [0,1] if strategy="mean").
    - "random": pick one grader uniformly per sample (includes empty masks).
    - "fixed":  pick the provided grader index for all samples.
    - "mean":   average over available grader channels (float in [0,1]).
    """
    assert masks.ndim == 4 and masks.size(1) == 4, "masks must be [B,4,H,W]"
    B, _, H, W = masks.shape
    device = masks.device
    info: Dict[str, Any] = {}

    if strategy == "random":
        if rng is None:
            rng = torch.Generator(device=device)
            rng.manual_seed(12345)
        idx = torch.randint(low=0, high=4, size=(B,), generator=rng, device=device)  # [B]
        gather_idx = idx.view(B, 1, 1, 1).expand(B, 1, H, W)  # [B,1,H,W]
        targets = torch.gather(masks.float(), dim=1, index=gather_idx)  # [B,1,H,W]
        info["chosen_indices"] = idx.tolist()

    elif strategy == "fixed":
        assert fixed_index is not None and 0 <= fixed_index < 4, "fixed_index must be in {0,1,2,3}"
        targets = masks[:, fixed_index:fixed_index+1, :, :].float()
        info["chosen_indices"] = [fixed_index] * B

    elif strategy == "mean":
        # average across 4 graders; result in [0,1]
        targets = masks.float().mean(dim=1, keepdim=True)
        info["chosen_indices"] = ["mean"] * B
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return targets, info


def reconstruction_loss_from_logits(
    logits: Tensor,                # [B,1,H,W] float
    masks: Tensor,                 # [B,4,H,W] uint8
    pad_mask: Tensor,              # [B,H,W] bool
    strategy: str = "random",      # "random" | "fixed" | "mean"
    fixed_index: Optional[int] = None,
    use_topk: bool = False,
    k_frac: float = 0.02,
    pos_weight: Optional[Tensor] = None,
    rng: Optional[torch.Generator] = None,
) -> Tuple[Tensor, Dict[str, Any]]:
    """
    Select targets according to a strategy and compute masked reconstruction loss.
    Returns (loss, info_dict).
    """
    assert logits.ndim == 4 and logits.size(1) == 1, "logits must be [B,1,H,W]"
    assert masks.ndim == 4 and masks.size(1) == 4, "masks must be [B,4,H,W]"
    assert pad_mask.ndim == 3, "pad_mask must be [B,H,W]"

    targets, info = select_targets_from_graders(
        masks=masks,
        strategy=strategy,
        fixed_index=fixed_index,
        rng=rng,
    )

    if use_topk:
        loss = masked_topk_bce_with_logits(
            logits=logits,
            targets=targets,
            pad_mask=pad_mask,
            k_frac=k_frac,
            pos_weight=pos_weight,
        )
    else:
        loss = masked_bce_with_logits(
            logits=logits,
            targets=targets,
            pad_mask=pad_mask,
            pos_weight=pos_weight,
        )

    info.update({
        "strategy": strategy,
        "use_topk": use_topk,
        "k_frac": k_frac if use_topk else 0.0,
    })
    return loss, info
