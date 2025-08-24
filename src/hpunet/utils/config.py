from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional, Set


_DEFAULTS: Dict[str, Any] = {
    "seed": 42,
    "total_steps": 240000,
    "batch_size": 32,
    "lr": 1e-4,
    "lr_milestones": [160000, 200000],
    "lr_gamma": 0.1,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "use_topk": False,
    "k_frac": 0.02,
    "recon_strategy": "random",  # "random" | "fixed" | "mean"
    "fixed_index": None,
    "eval_every_steps": 5000,
    "ckpt_every_steps": 10000,
    "num_workers": 4,
    "augment": True,
    "geco": None,  # e.g., {"kappa":0.05, "alpha":0.99, "lambda_init":1.0}
    "pos_weight": None,           # "auto" | float | None
    "pos_weight_clip": 20.0
}

_ALLOWED_KEYS: Set[str] = set(_DEFAULTS.keys())


def _merge_defaults(user: Dict[str, Any]) -> Dict[str, Any]:
    cfg = _DEFAULTS.copy()
    for k, v in user.items():
        if k not in _ALLOWED_KEYS:
            # ignore silently to keep this minimal; could also raise
            continue
        cfg[k] = v
    return cfg


def load_config(path: str | Path, overrides: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """
    Load a JSON config and merge with defaults. Unknown keys are ignored.
    Returns a SimpleNamespace for dot-access (cfg.lr, cfg.batch_size, ...).
    """
    path = Path(path)
    with path.open("r") as f:
        user_cfg = json.load(f)
    cfg_dict = _merge_defaults(user_cfg)
    if overrides:
        cfg_dict = _merge_defaults({**cfg_dict, **overrides})
    return SimpleNamespace(**cfg_dict)
