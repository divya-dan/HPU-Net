from __future__ import annotations
from typing import Iterable, Mapping, Any
import torch
from torch import optim

def make_optimizer(params: Iterable[torch.nn.Parameter], cfg) -> optim.Optimizer:
    if cfg.optimizer.lower() == "adam":
        return optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer.lower() == "sgd":
        return optim.SGD(params, lr=cfg.lr, momentum=0.9, nesterov=True, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")

def make_scheduler(optimizer: optim.Optimizer, cfg):
    # Step milestones are in "steps", so call .step() every iteration.
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)
