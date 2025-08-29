"""
Paper-faithful sPU-Net (Probabilistic U-Net) for LIDC
-----------------------------------------------------------------
- 5-scale standard U-Net (no BatchNorm), 3×(3×3 conv + ReLU) per scale
- Bilinear downsampling/upsampling between scales
- Separate prior and posterior networks that mirror the encoder
- Single global latent vector z ∈ R^6
- Latent is broadcast and concatenated with decoder features
- Combiner: three 1×1 conv layers → logits (no sigmoid)
- Weight init: orthogonal weights, truncated-normal biases (σ=1e-3)
- Forward returns (logits, info) with per-sample KL (sum over z-dims)
"""
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional

import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def _truncated_normal_(tensor: torch.Tensor, std: float = 1e-3, a: float = -2.0, b: float = 2.0) -> None:
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
        tensor.clamp_(a * std, b * std)


def gaussian_kl(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
    return kl.sum(dim=1)


# -----------------------------------------------------------------------------
# Paper-faithful Conv Blocks (3×3 conv + ReLU, bias=True)
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch: int, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)
        self.enc5 = ConvBlock(base*8, base*16)
        # (downsampling via bilinear interpolation in forward)
        self.channels = [base, base*2, base*4, base*8, base*16]

    def forward(self, x: torch.Tensor):
        e1 = self.enc1(x)
        e2 = self.enc2(F.interpolate(e1, scale_factor=0.5, mode="bilinear", align_corners=False))
        e3 = self.enc3(F.interpolate(e2, scale_factor=0.5, mode="bilinear", align_corners=False))
        e4 = self.enc4(F.interpolate(e3, scale_factor=0.5, mode="bilinear", align_corners=False))
        e5 = self.enc5(F.interpolate(e4, scale_factor=0.5, mode="bilinear", align_corners=False))
        return e1, e2, e3, e4, e5


class Decoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        C1, C2, C3, C4, C5 = channels
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = ConvBlock(C5 + C4, C4)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = ConvBlock(C4 + C3, C3)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = ConvBlock(C3 + C2, C2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = ConvBlock(C2 + C1, C1)
        self.out_ch = C1

    def forward(self, e1, e2, e3, e4, e5):
        d4 = self.dec4(torch.cat([self.up4(e5), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return d1


# -----------------------------------------------------------------------------
# Prior & Posterior Networks
# -----------------------------------------------------------------------------

class PriorNet(nn.Module):
    def __init__(self, in_ch: int, base: int, z_dim: int):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch, base=base)
        C1, C2, C3, C4, C5 = self.encoder.channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mu_head = nn.Conv2d(C5, z_dim, kernel_size=1, bias=True)
        self.logvar_head = nn.Conv2d(C5, z_dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        *_, e5 = self.encoder(x)
        h = self.global_pool(e5)
        mu = self.mu_head(h).squeeze(-1).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1).squeeze(-1).clamp(-10.0, 10.0)
        return mu, logvar


class PosteriorNet(nn.Module):
    def __init__(self, in_ch: int, base: int, z_dim: int):
        super().__init__()
        self.encoder = Encoder(in_ch=in_ch + 1, base=base)
        C1, C2, C3, C4, C5 = self.encoder.channels
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mu_head = nn.Conv2d(C5, z_dim, kernel_size=1, bias=True)
        self.logvar_head = nn.Conv2d(C5, z_dim, kernel_size=1, bias=True)

    def forward(self, x_and_y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        *_, e5 = self.encoder(x_and_y)
        h = self.global_pool(e5)
        mu = self.mu_head(h).squeeze(-1).squeeze(-1)
        logvar = self.logvar_head(h).squeeze(-1).squeeze(-1).clamp(-10.0, 10.0)
        return mu, logvar


# -----------------------------------------------------------------------------
# Combiner
# -----------------------------------------------------------------------------

class Combiner(nn.Module):
    def __init__(self, dec_ch: int, z_dim: int):
        super().__init__()
        self.combine = nn.Sequential(
            nn.Conv2d(dec_ch + z_dim, dec_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, dec_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, 1, kernel_size=1, bias=True),
        )

    def forward(self, dec_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dec_feat.shape
        z_b = z.view(B, -1, 1, 1).expand(B, z.size(1), H, W)
        x = torch.cat([dec_feat, z_b], dim=1)
        return self.combine(x)


# -----------------------------------------------------------------------------
# sPU-Net main module
# -----------------------------------------------------------------------------

class sPUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32, z_dim: int = 6):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = Encoder(in_ch=in_ch, base=base)
        self.decoder = Decoder(self.encoder.channels)
        self.prior = PriorNet(in_ch=in_ch, base=base, z_dim=z_dim)
        self.post = PosteriorNet(in_ch=in_ch, base=base, z_dim=z_dim)
        self.combiner = Combiner(dec_ch=self.decoder.out_ch, z_dim=z_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    _truncated_normal_(m.bias, std=1e-3)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    _truncated_normal_(m.bias, std=1e-3)

    def forward(self, x: torch.Tensor, y_target: Optional[torch.Tensor] = None, sample_posterior: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        e1, e2, e3, e4, e5 = self.encoder(x)
        dec_feat = self.decoder(e1, e2, e3, e4, e5)

        mu_p, logvar_p = self.prior(x)

        mu_q = logvar_q = None
        if (y_target is not None) and sample_posterior:
            xy = torch.cat([x, y_target], dim=1)
            mu_q, logvar_q = self.post(xy)
            std_q = torch.exp(0.5 * logvar_q)
            z = mu_q + torch.randn_like(std_q) * std_q
        else:
            std_p = torch.exp(0.5 * logvar_p)
            z = mu_p + torch.randn_like(std_p) * std_p

        logits = self.combiner(dec_feat, z)

        if (mu_q is not None) and (logvar_q is not None):
            kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)
        else:
            kl = torch.zeros(x.size(0), device=x.device)

        info: Dict[str, Any] = {
            "mu_p": mu_p, "logvar_p": logvar_p,
            "mu_q": mu_q, "logvar_q": logvar_q,
            "z": z,
            "kl": kl,
        }
        return logits, info
