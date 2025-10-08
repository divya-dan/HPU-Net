from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
import torch
from torch import Tensor


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