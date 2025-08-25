from __future__ import annotations
from dataclasses import dataclass
from mimetypes import init
from typing import Optional, Dict, Any
import torch

@dataclass
class GECOConfig:
    kappa: float = 0.05     
    alpha: float = 0.99     
    lambda_init: float = 1.0
    min_log_lambda: float = -8.0
    max_log_lambda: float = 8.0
    step_size: float = 0.01  # CHANGE: was 0.1, should be 0.01

class GECO:
    """
    GECO: maintain Lagrange multiplier λ >= 0 to enforce E[recon] ≈ κ.
    Constraint: C = recon - kappa. EMA: C_bar_t = α C_bar_{t-1} + (1-α) C_t
    Update: log λ <- log λ + step_size * C_bar_t
    L_total (for backprop): KL + λ * (recon - κ)
    """
    def __init__(self, cfg: GECOConfig):
        self.cfg = cfg
        # work in log-space for numerical stability
        init = max(1e-8, float(cfg.lambda_init))
        self.log_lambda = torch.tensor(float(torch.log(torch.tensor(init))), dtype=torch.float32)
        self.ema_c = torch.tensor(0.0, dtype=torch.float32)
        self._device = torch.device("cpu")

    def to(self, device: torch.device):
        self._device = device
        self.log_lambda = self.log_lambda.to(device)
        self.ema_c = self.ema_c.to(device)
        return self

    @property
    def lam(self) -> torch.Tensor:
        return torch.exp(self.log_lambda)

    def step(self, recon: torch.Tensor) -> Dict[str, Any]:
        """
        recon: scalar tensor (batch-mean reconstruction loss)
        Returns dict of current λ, C, C_bar
        """
        # constraint: C = recon - kappa
        C = recon.detach() - self.cfg.kappa
        # EMA over time
        self.ema_c = self.cfg.alpha * self.ema_c + (1.0 - self.cfg.alpha) * C
        # log-lambda update
        self.log_lambda = self.log_lambda + self.cfg.step_size * self.ema_c
        self.log_lambda = torch.clamp(self.log_lambda, self.cfg.min_log_lambda, self.cfg.max_log_lambda)
        return {
            "lambda": float(self.lam.item()),
            "C": float(C.item()),
            "C_bar": float(self.ema_c.item()),
        }

    def lagrangian(self, recon: torch.Tensor, kl: torch.Tensor) -> torch.Tensor:
        """
        L = KL + λ * (recon - κ)
        Both recon and KL should be scalar batch-means.
        """
        return kl + self.lam * (recon - self.cfg.kappa)
