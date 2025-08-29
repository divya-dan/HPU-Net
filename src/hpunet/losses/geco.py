from __future__ import annotations
from dataclasses import dataclass
from mimetypes import init
from typing import Optional, Dict, Any
import torch


@dataclass
class GECOConfig:
    kappa: float = 0.05     # target reconstruction loss
    alpha: float = 0.99     # EMA decay factor
    lambda_init: float = 1.0
    min_log_lambda: float = -8.0
    max_log_lambda: float = 8.0
    step_size: float = 0.01  # was 0.1, but 0.01 works better


class GECO:
    """
    GECO optimizer - maintains lagrange multiplier to keep reconstruction loss around kappa
    
    The idea is: constraint C = recon - kappa, we want this to be ~0
    EMA: C_bar_t = alpha * C_bar_{t-1} + (1-alpha) * C_t  
    Update: log lambda <- log lambda + step_size * C_bar_t
    Total loss for backprop: KL + lambda * (recon - kappa)
    """
    def __init__(self, cfg: GECOConfig):
        self.cfg = cfg
        # work in log space to avoid numerical issues
        init = max(1e-8, float(cfg.lambda_init))
        self.log_lambda = torch.tensor(float(torch.log(torch.tensor(init))), dtype=torch.float32)
        self.ema_c = torch.tensor(0.0, dtype=torch.float32)
        self._device = torch.device("cpu")

    def to(self, device: torch.device):
        """move tensors to device"""
        self._device = device
        self.log_lambda = self.log_lambda.to(device)
        self.ema_c = self.ema_c.to(device)
        return self

    @property
    def lam(self) -> torch.Tensor:
        """current lambda value (exponentiated from log space)"""
        return torch.exp(self.log_lambda)

    def step(self, recon: torch.Tensor) -> Dict[str, Any]:
        """
        update the lagrange multiplier based on current reconstruction loss
        recon: scalar tensor (batch-mean reconstruction loss)
        returns dict with current lambda, C, C_bar values
        """
        # constraint: C = recon - kappa (we want this close to 0)
        C = recon.detach() - self.cfg.kappa
        
        # update EMA of constraint violation
        self.ema_c = self.cfg.alpha * self.ema_c + (1.0 - self.cfg.alpha) * C
        
        # update log-lambda based on EMA
        self.log_lambda = self.log_lambda + self.cfg.step_size * self.ema_c
        self.log_lambda = torch.clamp(self.log_lambda, self.cfg.min_log_lambda, self.cfg.max_log_lambda)
        
        return {
            "lambda": float(self.lam.item()),
            "C": float(C.item()),
            "C_bar": float(self.ema_c.item()),
        }

    def lagrangian(self, recon: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        """
        compute the lagrangian: L = KL + lambda * (recon - kappa)
        both recon and KL should be scalar batch-means
        """
        return kl + self.lam * (recon - self.cfg.kappa)