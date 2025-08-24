from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
import torch
from torch import nn
import torch.nn.functional as F

from .unet_blocks import UNetEncoder, UNetDecoder


def gaussian_kl_spatial(mu_q: torch.Tensor, logvar_q: torch.Tensor,
                        mu_p: torch.Tensor, logvar_p: torch.Tensor) -> torch.Tensor:
    """
    KL per-sample for spatial Gaussians.
    Inputs: [B,C,H,W]. Return [B] (sum over C,H,W).
    """
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl_map = 0.5 * ((logvar_p - logvar_q) + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
    return kl_map.flatten(1).sum(dim=1)  # [B]


class GaussianParam2d(nn.Module):
    """Predicts spatial mean/logvar from feature maps."""
    def __init__(self, in_ch: int, z_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2 * z_ch, 3, padding=1)
        )
        self.z_ch = z_ch

    def forward(self, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(feat)
        mu, logvar = torch.split(h, self.z_ch, dim=1)
        logvar = logvar.clamp(-10.0, 10.0)  # out-of-place to avoid view mutation error
        return mu, logvar


class InjectLatent(nn.Module):
    """Project latent map z (Cz,H,W) to feature channels (Cf,H,W) and add."""
    def __init__(self, z_ch: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(z_ch, feat_ch, 1)

    def forward(self, feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return feat + self.proj(z)


class HPUNet(nn.Module):
    """
    Hierarchical Probabilistic U-Net (simplified 4-scale version).
    Spatial latents at resolutions: 8x8, 16x16, 32x32, 64x64 injected into the decoder.
    Prior pθ(z_s|x, z_{>s}) and Posterior qφ(z_s|x,y, z_{>s}) are parameterized
    from features at the corresponding scale; posterior only used during training.
    """
    def __init__(self, in_ch: int = 1, base: int = 32, z_ch: int = 8):
        super().__init__()
        # Shared image encoder/decoder
        self.enc_x = UNetEncoder(in_ch=in_ch, base=base)
        self.dec_x = UNetDecoder(self.enc_x.channels)
        C1, C2, C3, C4, C5 = self.enc_x.channels  # 128..8 resolutions

        # Posterior encoder/decoder on [x,y]
        self.enc_xy = UNetEncoder(in_ch=in_ch + 1, base=base)
        self.dec_xy = UNetDecoder(self.enc_xy.channels)

        # Prior/posterior param heads per scale (coarse->fine)
        # s8 uses features at 8x8 (C5), s16 at decoder stage C4, s32 at C3, s64 at C2
        self.prior_s8   = GaussianParam2d(C5, z_ch)
        self.prior_s16  = GaussianParam2d(C4, z_ch)
        self.prior_s32  = GaussianParam2d(C3, z_ch)
        self.prior_s64  = GaussianParam2d(C2, z_ch)

        self.post_s8    = GaussianParam2d(C5, z_ch)
        self.post_s16   = GaussianParam2d(C4, z_ch)
        self.post_s32   = GaussianParam2d(C3, z_ch)
        self.post_s64   = GaussianParam2d(C2, z_ch)

        # Injection projections for x-decoder path
        self.inj_s8  = InjectLatent(z_ch, C5)
        self.inj_s16 = InjectLatent(z_ch, C4)
        self.inj_s32 = InjectLatent(z_ch, C3)
        self.inj_s64 = InjectLatent(z_ch, C2)

        # Injection projections for xy posterior decoder (to condition next levels)
        self.pin_s8  = InjectLatent(z_ch, C5)
        self.pin_s16 = InjectLatent(z_ch, C4)
        self.pin_s32 = InjectLatent(z_ch, C3)
        self.pin_s64 = InjectLatent(z_ch, C2)

        # Final prediction head on 128x128 decoder output
        self.head = nn.Sequential(
            nn.Conv2d(C1, C1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C1, 1, 1)
        )

    def forward(self, x: torch.Tensor, y_target: Optional[torch.Tensor] = None,
                sample_posterior: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        x: [B,1,128,128]; y_target: [B,1,128,128] if training with posterior
        Returns logits [B,1,128,128] and info dict with KLs.
        """
        B = x.size(0)
        # 1) Encode image
        e1, e2, e3, e4, e5 = self.enc_x(x)   # 128, 64, 32, 16, 8

        # 2) If posterior path needed, encode [x,y] and prep decoder features
        #    Always initialize pdec* so prior-only path won't reference undefined vars.
        pdec16 = pdec32 = pdec64 = None
        if (y_target is not None) and sample_posterior:
            xy = torch.cat([x, y_target], dim=1)
            pe1, pe2, pe3, pe4, pe5 = self.enc_xy(xy)
            # placeholders; we’ll build posterior decoder features as we go
            # (already initialized above)
        else:
            pe1 = pe2 = pe3 = pe4 = pe5 = None

        # Containers
        kl_terms: List[torch.Tensor] = []
        zs: List[torch.Tensor] = []

        # ---- Scale s=8 (coarsest, 8x8) ----
        mu_p8, lv_p8 = self.prior_s8(e5)
        if pe5 is not None:
            mu_q8, lv_q8 = self.post_s8(pe5)
            std_q = torch.exp(0.5 * lv_q8)
            z8 = mu_q8 + torch.randn_like(std_q) * std_q
            kl_terms.append(gaussian_kl_spatial(mu_q8, lv_q8, mu_p8, lv_p8))
        else:
            std_p = torch.exp(0.5 * lv_p8)
            z8 = mu_p8 + torch.randn_like(std_p) * std_p
        zs.append(z8)

        # inject z8 into e5 and (if posterior) into pe5
        e5_inj = self.inj_s8(e5, z8)
        if pe5 is not None:
            pe5_inj = self.pin_s8(pe5, z8)

        # up to 16x16 on both paths
        d4 = self.dec_x.up4(e5_inj, e4)       # 16x16, C4
        if pe5 is not None:
            pdec16 = self.dec_xy.up4(pe5_inj, pe4)  # 16x16, C4

        # ---- Scale s=16 ----
        mu_p16, lv_p16 = self.prior_s16(d4)
        if pdec16 is not None:
            mu_q16, lv_q16 = self.post_s16(pdec16)
            std_q = torch.exp(0.5 * lv_q16)
            z16 = mu_q16 + torch.randn_like(std_q) * std_q
            kl_terms.append(gaussian_kl_spatial(mu_q16, lv_q16, mu_p16, lv_p16))
        else:
            std_p = torch.exp(0.5 * lv_p16)
            z16 = mu_p16 + torch.randn_like(std_p) * std_p
        zs.append(z16)

        d4 = self.inj_s16(d4, z16)
        if pdec16 is not None:
            pdec16 = self.pin_s16(pdec16, z16)

        d3 = self.dec_x.up3(d4, e3)           # 32x32, C3
        if pdec16 is not None:
            pdec32 = self.dec_xy.up3(pdec16, pe3)

        # ---- Scale s=32 ----
        mu_p32, lv_p32 = self.prior_s32(d3)
        if pdec32 is not None:
            mu_q32, lv_q32 = self.post_s32(pdec32)
            std_q = torch.exp(0.5 * lv_q32)
            z32 = mu_q32 + torch.randn_like(std_q) * std_q
            kl_terms.append(gaussian_kl_spatial(mu_q32, lv_q32, mu_p32, lv_p32))
        else:
            std_p = torch.exp(0.5 * lv_p32)
            z32 = mu_p32 + torch.randn_like(std_p) * std_p
        zs.append(z32)

        d3 = self.inj_s32(d3, z32)
        if pdec32 is not None:
            pdec32 = self.pin_s32(pdec32, z32)

        d2 = self.dec_x.up2(d3, e2)           # 64x64, C2
        if pdec32 is not None:
            pdec64 = self.dec_xy.up2(pdec32, pe2)

        # ---- Scale s=64 ----
        mu_p64, lv_p64 = self.prior_s64(d2)
        if pdec64 is not None:
            mu_q64, lv_q64 = self.post_s64(pdec64)
            std_q = torch.exp(0.5 * lv_q64)
            z64 = mu_q64 + torch.randn_like(std_q) * std_q
            kl_terms.append(gaussian_kl_spatial(mu_q64, lv_q64, mu_p64, lv_p64))
        else:
            std_p = torch.exp(0.5 * lv_p64)
            z64 = mu_p64 + torch.randn_like(std_p) * std_p
        zs.append(z64)

        d2 = self.inj_s64(d2, z64)

        # finish decode to 128x128
        d1 = self.dec_x.up1(d2, e1)           # 128x128, C1
        logits = self.head(d1)                # [B,1,128,128]

        # Collect KL (sum over scales, mean over batch)
        if len(kl_terms) > 0:
            kl_stack = torch.stack(kl_terms, dim=0)  # [S,B]
            kl_per_sample = kl_stack.sum(dim=0)      # [B]
            kl_mean = kl_per_sample.mean()
        else:
            kl_mean = torch.zeros(B, device=x.device).mean()

        info: Dict[str, Any] = {
            "kl": kl_mean,
            "kl_per_level": [k.detach() for k in kl_terms],
            "z_shapes": [tuple(z.shape) for z in zs],
        }
        return logits, info
