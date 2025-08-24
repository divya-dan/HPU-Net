from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
import torch.nn.functional as F

from .unet_blocks import UNetEncoder, UNetDecoder


def gaussian_kl(mu_q, logvar_q, mu_p, logvar_p) -> torch.Tensor:
    """
    KL( N(mu_q, sigma_q^2) || N(mu_p, sigma_p^2) ) per sample.
    Returns [B] (sum over latent dims).
    """
    # all shape: [B, Z]
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        (logvar_p - logvar_q)
        + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8)
        - 1.0
    )
    return kl.sum(dim=1)


class LatentHead(nn.Module):
    """Predicts mean and logvar from a feature tensor."""
    def __init__(self, in_ch: int, z_dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.mu = nn.Linear(in_ch, z_dim)
        self.logvar = nn.Linear(in_ch, z_dim)

    def forward(self, feat) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv(feat).flatten(1)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp_(-10.0, 10.0)  # stabilize
        return mu, logvar


class CombinerHead(nn.Module):
    """Combine decoder features with broadcast latent and produce logits."""
    def __init__(self, in_ch: int, z_dim: int):
        super().__init__()
        self.proj_z = nn.Conv2d(z_dim, in_ch, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, kernel_size=1)  # logits
        )

    def forward(self, dec_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # z: [B, Z] -> [B, Z, H, W]
        B, C, H, W = dec_feat.shape
        z_ = z.view(B, -1, 1, 1).expand(B, -1, H, W)
        z_proj = self.proj_z(z_)
        x = torch.cat([dec_feat, z_proj], dim=1)
        return self.fuse(x)


class ProbUNet(nn.Module):
    """
    Minimal Probabilistic U-Net (global latent).
    Encoder-decoder on X, prior on X, posterior on concat(X,Y).
    """
    def __init__(self, in_ch: int = 1, base: int = 32, z_dim: int = 6):
        super().__init__()
        # shared image encoder/decoder
        self.encoder = UNetEncoder(in_ch=in_ch, base=base)
        self.decoder = UNetDecoder(self.encoder.channels)
        # prior p(z|x) from deepest feature
        C1, C2, C3, C4, C5 = self.encoder.channels
        self.prior = LatentHead(C5, z_dim)
        # posterior q(z|x,y) gets image+mask as 2 channels
        self.post_encoder = UNetEncoder(in_ch=in_ch + 1, base=base)
        self.posterior = LatentHead(self.post_encoder.channels[-1], z_dim)
        # combine latent with decoder features
        self.combiner = CombinerHead(self.decoder.out_ch, z_dim)

    def forward(self, x: torch.Tensor, y_target: Optional[torch.Tensor] = None, sample_posterior: bool = True
                ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        x: [B,1,H,W]
        y_target: [B,1,H,W] in {0,1} if training with posterior
        sample_posterior: if True and y_target provided -> sample from q(z|x,y), else from p(z|x)
        Returns: logits [B,1,H,W], info dict (mu/logvar for p & q, z, KL)
        """
        # Encode image
        e1, e2, e3, e4, e5 = self.encoder(x)
        # Prior
        mu_p, logvar_p = self.prior(e5)  # [B,Z]
        # Decoder features
        dec_feat = self.decoder(e1, e2, e3, e4, e5)  # [B,C,128,128]

        mu_q = logvar_q = None
        if (y_target is not None) and sample_posterior:
            # Posterior over z | x,y
            xy = torch.cat([x, y_target], dim=1)
            pe1, pe2, pe3, pe4, pe5 = self.post_encoder(xy)
            mu_q, logvar_q = self.posterior(pe5)  # [B,Z]
            # reparameterize
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + eps * std_q
        else:
            # sample from prior
            std_p = torch.exp(0.5 * logvar_p)
            eps = torch.randn_like(std_p)
            z = mu_p + eps * std_p

        logits = self.combiner(dec_feat, z)  # [B,1,H,W]

        info: Dict[str, Any] = {
            "mu_p": mu_p, "logvar_p": logvar_p, "z": z,
            "mu_q": mu_q, "logvar_q": logvar_q,
        }
        if (mu_q is not None) and (logvar_q is not None):
            kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)  # [B]
        else:
            # if no posterior (pure prior sampling), KL undefined -> zero
            kl = torch.zeros(x.size(0), device=x.device)
        info["kl"] = kl
        return logits, info
