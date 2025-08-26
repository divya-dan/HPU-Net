# hpu_net.py - COMPLETE 8-SCALE IMPLEMENTATION
import torch
from torch import nn
import torch.nn.functional as F
from .unet_blocks import ResStack, ResDown, ResUp

def gaussian_kl_spatial(mu_q, lv_q, mu_p, lv_p) -> torch.Tensor:
    """KL(q||p) for diagonal Gaussians over (C,H,W), return [B]"""
    var_q, var_p = torch.exp(lv_q), torch.exp(lv_p)
    kl = 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / (var_p + 1e-8) - 1.0)
    return kl.flatten(1).sum(1)  # [B]

class GaussianParam2d(nn.Module):
    """Predicts mean and log-variance for 2D Gaussian distributions"""
    def __init__(self, in_ch: int, z_ch: int):
        super().__init__()
        self.z_ch = z_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2 * z_ch, 3, padding=1)
        )

    def forward(self, feat):
        h = self.net(feat)                         # [B, 2*z_ch, H, W]
        mu, lv = torch.split(h, self.z_ch, dim=1)  # split channels
        lv = torch.clamp(lv, -10.0, 10.0)          # stabilize log-variance
        return mu, lv

class InjectLatent(nn.Module):
    """Upsample z to feat size, project, then add to feat."""
    def __init__(self, z_ch: int, feat_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(z_ch, feat_ch, 1)
    
    def forward(self, feat, z):
        if z.shape[2:] != feat.shape[2:]:
            z = F.interpolate(z, size=feat.shape[2:], mode="nearest")
        return feat + self.proj(z)

# ============================================================================
# 8-SCALE ResUNet ENCODER/DECODER
# ============================================================================

def _cap(c):
    """Cap channels at 192 as per paper spec"""
    return min(c, 192)

class ResUNet8ScaleEncoder(nn.Module):
    """
    8-scale ResUNet encoder: 128→64→32→16→8→4→2→1
    Base=24, channels: (24, 48, 96, 192, 192, 192, 192, 192)
    """
    def __init__(self, in_ch: int = 1, base: int = 24, n_blocks: int = 3):
        super().__init__()
        
        # Channel progression as per paper: base, 2*base, 4*base, 8*base, then capped at 192
        C1 = _cap(base)          # 24
        C2 = _cap(base * 2)      # 48  
        C3 = _cap(base * 4)      # 96
        C4 = _cap(base * 8)      # 192 (= min(192, 192))
        C5 = _cap(base * 16)     # 192 (capped)
        C6 = _cap(base * 32)     # 192 (capped)
        C7 = _cap(base * 64)     # 192 (capped)
        C8 = _cap(base * 128)    # 192 (capped)

        
        self.enc1  = ResStack(in_ch, C1, n_blocks)  # 128x128
        self.down1 = ResDown(C1, C2, n_blocks)      # 64x64
        self.down2 = ResDown(C2, C3, n_blocks)      # 32x32  
        self.down3 = ResDown(C3, C4, n_blocks)      # 16x16
        self.down4 = ResDown(C4, C5, n_blocks)      # 8x8
        self.down5 = ResDown(C5, C6, n_blocks)      # 4x4
        self.down6 = ResDown(C6, C7, n_blocks)      # 2x2
        self.down7 = ResDown(C7, C8, n_blocks)      # 1x1 (global)

        self.channels = (C1, C2, C3, C4, C5, C6, C7, C8)

    def forward(self, x):
        e1 = self.enc1(x)     # 128x128
        e2 = self.down1(e1)   # 64x64
        e3 = self.down2(e2)   # 32x32
        e4 = self.down3(e3)   # 16x16
        e5 = self.down4(e4)   # 8x8
        e6 = self.down5(e5)   # 4x4
        e7 = self.down6(e6)   # 2x2
        e8 = self.down7(e7)   # 1x1
        return e1, e2, e3, e4, e5, e6, e7, e8

class ResUNet8ScaleDecoder(nn.Module):
    """8-scale ResUNet decoder: 1→2→4→8→16→32→64→128"""
    def __init__(self, chans: tuple, n_blocks: int = 3):
        super().__init__()
        C1, C2, C3, C4, C5, C6, C7, C8 = chans
        
        self.up7 = ResUp(C8, C7, C7, n_blocks)  # 1x1 + 2x2 -> 2x2
        self.up6 = ResUp(C7, C6, C6, n_blocks)  # 2x2 + 4x4 -> 4x4
        self.up5 = ResUp(C6, C5, C5, n_blocks)  # 4x4 + 8x8 -> 8x8
        self.up4 = ResUp(C5, C4, C4, n_blocks)  # 8x8 + 16x16 -> 16x16
        self.up3 = ResUp(C4, C3, C3, n_blocks)  # 16x16 + 32x32 -> 32x32
        self.up2 = ResUp(C3, C2, C2, n_blocks)  # 32x32 + 64x64 -> 64x64
        self.up1 = ResUp(C2, C1, C1, n_blocks)  # 64x64 + 128x128 -> 128x128
        self.out_ch = C1

    def forward(self, e1, e2, e3, e4, e5, e6, e7, e8):
        d7 = self.up7(e8, e7)  # 1x1 -> 2x2
        d6 = self.up6(d7, e6)  # 2x2 -> 4x4
        d5 = self.up5(d6, e5)  # 4x4 -> 8x8
        d4 = self.up4(d5, e4)  # 8x8 -> 16x16
        d3 = self.up3(d4, e3)  # 16x16 -> 32x32
        d2 = self.up2(d3, e2)  # 32x32 -> 64x64
        d1 = self.up1(d2, e1)  # 64x64 -> 128x128
        return d1, d2, d3, d4, d5, d6, d7  # Return all decoder features for latent injection

# ============================================================================
# MAIN HPUNet MODEL
# ============================================================================

class HPUNet(nn.Module):
    """
    8-scale Hierarchical Probabilistic U-Net for LIDC dataset.
    Latent scales at: 1x1, 2x2, 4x4, 8x8, 16x16, 32x32, 64x64, 128x128.
    Uses ResUNet backbone with base=24 channels and 3 pre-activated residual blocks per scale.
    """
    def __init__(self, in_ch: int = 1, base: int = 24, z_ch: int = 1, n_blocks: int = 3):
        super().__init__()
        self.z_ch = z_ch

        # 8-scale ResUNet encoders (prior and posterior)
        self.enc_x  = ResUNet8ScaleEncoder(in_ch=in_ch,     base=base, n_blocks=n_blocks)
        self.enc_xy = ResUNet8ScaleEncoder(in_ch=in_ch + 1, base=base, n_blocks=n_blocks)
        self.dec    = ResUNet8ScaleDecoder(self.enc_x.channels, n_blocks=n_blocks)
        
        # Channel configuration: (24, 48, 96, 192, 192, 192, 192, 192)
        C1, C2, C3, C4, C5, C6, C7, C8 = self.enc_x.channels

        # Prior heads at each latent scale
        self.prior_s1   = GaussianParam2d(C8, z_ch)  # 1x1 (global)
        self.prior_s2   = GaussianParam2d(C7, z_ch)  # 2x2  
        self.prior_s4   = GaussianParam2d(C6, z_ch)  # 4x4
        self.prior_s8   = GaussianParam2d(C5, z_ch)  # 8x8
        self.prior_s16  = GaussianParam2d(C4, z_ch)  # 16x16
        self.prior_s32  = GaussianParam2d(C3, z_ch)  # 32x32
        self.prior_s64  = GaussianParam2d(C2, z_ch)  # 64x64
        self.prior_s128 = GaussianParam2d(C1, z_ch)  # 128x128

        # Posterior heads at each latent scale
        self.post_s1    = GaussianParam2d(C8, z_ch)
        self.post_s2    = GaussianParam2d(C7, z_ch)
        self.post_s4    = GaussianParam2d(C6, z_ch)
        self.post_s8    = GaussianParam2d(C5, z_ch)
        self.post_s16   = GaussianParam2d(C4, z_ch)
        self.post_s32   = GaussianParam2d(C3, z_ch)
        self.post_s64   = GaussianParam2d(C2, z_ch)
        self.post_s128  = GaussianParam2d(C1, z_ch)

        # Latent injectors for prior decoder
        self.inj_s1  = InjectLatent(z_ch, C8)  # inject at 1x1
        self.inj_s2  = InjectLatent(z_ch, C7)  # inject at 2x2
        self.inj_s4  = InjectLatent(z_ch, C6)  # inject at 4x4
        self.inj_s8  = InjectLatent(z_ch, C5)  # inject at 8x8
        self.inj_s16 = InjectLatent(z_ch, C4)  # inject at 16x16
        self.inj_s32 = InjectLatent(z_ch, C3)  # inject at 32x32
        self.inj_s64 = InjectLatent(z_ch, C2)  # inject at 64x64
        self.inj_s128 = InjectLatent(z_ch, C1) # inject at 128x128

        # Latent injectors for posterior decoder (for conditioning next scale)
        self.pin_s1  = InjectLatent(z_ch, C8)
        self.pin_s2  = InjectLatent(z_ch, C7)
        self.pin_s4  = InjectLatent(z_ch, C6)
        self.pin_s8  = InjectLatent(z_ch, C5)
        self.pin_s16 = InjectLatent(z_ch, C4)
        self.pin_s32 = InjectLatent(z_ch, C3)
        self.pin_s64 = InjectLatent(z_ch, C2)
        self.pin_s128 = InjectLatent(z_ch, C1)

        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(C1, C1, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(C1, 1, 1)  # Binary segmentation
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights as per paper specification"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Orthogonal initialization with gain=1.0
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    # Truncated normal with stddev=0.001
                    nn.init.normal_(m.bias, mean=0.0, std=0.001)

    def forward(self, x: torch.Tensor,
                y_target: torch.Tensor | None = None,
                sample_posterior: bool = True):
        """
        Forward pass with hierarchical latent sampling.
        
        Args:
            x: Input image [B,1,H,W]
            y_target: Target segmentation [B,1,H,W] (for posterior sampling)
            sample_posterior: If True and y_target provided, sample from posterior
        
        Returns:
            logits: Prediction logits [B,1,H,W]
            info: Dict with KL_sum and all distribution parameters
        """
        B = x.size(0)
        
        # Encode input image
        e1, e2, e3, e4, e5, e6, e7, e8 = self.enc_x(x)

        # Encode image+target for posterior if available
        posterior_features = None
        if (y_target is not None) and sample_posterior:
            xy = torch.cat([x, y_target], dim=1)
            posterior_features = self.enc_xy(xy)

        info = {}
        KLs = []

        # =================================================================
        # HIERARCHICAL LATENT SAMPLING (8 SCALES)
        # =================================================================

        # ===================== Scale 1: 1x1 (Global, Deepest) =====================
        mu_p1, lv_p1 = self.prior_s1(e8)
        if posterior_features is not None:
            pe1, pe2, pe3, pe4, pe5, pe6, pe7, pe8 = posterior_features
            mu_q1, lv_q1 = self.post_s1(pe8)
            z1 = mu_q1 + torch.randn_like(mu_q1) * torch.exp(0.5 * lv_q1)
            KLs.append(gaussian_kl_spatial(mu_q1, lv_q1, mu_p1, lv_p1))
            # Condition posterior features for next scale
            pe8_conditioned = self.pin_s1(pe8, z1)
        else:
            z1 = mu_p1 + torch.randn_like(mu_p1) * torch.exp(0.5 * lv_p1)

        # Inject z1 and decode: 1x1 -> 2x2
        e8_with_z1 = self.inj_s1(e8, z1)
        d7 = self.dec.up7(e8_with_z1, e7)
        
        # Posterior path for next scale
        if posterior_features is not None:
            pd7 = self.dec.up7(pe8_conditioned, pe7)

        # ===================== Scale 2: 2x2 =====================
        mu_p2, lv_p2 = self.prior_s2(d7)
        if posterior_features is not None:
            mu_q2, lv_q2 = self.post_s2(pd7)
            z2 = mu_q2 + torch.randn_like(mu_q2) * torch.exp(0.5 * lv_q2)
            KLs.append(gaussian_kl_spatial(mu_q2, lv_q2, mu_p2, lv_p2))
            pe7_conditioned = self.pin_s2(pd7, z2)
        else:
            z2 = mu_p2 + torch.randn_like(mu_p2) * torch.exp(0.5 * lv_p2)

        d7_with_z2 = self.inj_s2(d7, z2)
        d6 = self.dec.up6(d7_with_z2, e6)
        
        if posterior_features is not None:
            pd6 = self.dec.up6(pe7_conditioned, pe6)

        # ===================== Scale 4: 4x4 =====================
        mu_p4, lv_p4 = self.prior_s4(d6)
        if posterior_features is not None:
            mu_q4, lv_q4 = self.post_s4(pd6)
            z4 = mu_q4 + torch.randn_like(mu_q4) * torch.exp(0.5 * lv_q4)
            KLs.append(gaussian_kl_spatial(mu_q4, lv_q4, mu_p4, lv_p4))
            pe6_conditioned = self.pin_s4(pd6, z4)
        else:
            z4 = mu_p4 + torch.randn_like(mu_p4) * torch.exp(0.5 * lv_p4)

        d6_with_z4 = self.inj_s4(d6, z4)
        d5 = self.dec.up5(d6_with_z4, e5)
        
        if posterior_features is not None:
            pd5 = self.dec.up5(pe6_conditioned, pe5)

        # ===================== Scale 8: 8x8 =====================
        mu_p8, lv_p8 = self.prior_s8(d5)
        if posterior_features is not None:
            mu_q8, lv_q8 = self.post_s8(pd5)
            z8 = mu_q8 + torch.randn_like(mu_q8) * torch.exp(0.5 * lv_q8)
            KLs.append(gaussian_kl_spatial(mu_q8, lv_q8, mu_p8, lv_p8))
            pe5_conditioned = self.pin_s8(pd5, z8)
        else:
            z8 = mu_p8 + torch.randn_like(mu_p8) * torch.exp(0.5 * lv_p8)

        d5_with_z8 = self.inj_s8(d5, z8)
        d4 = self.dec.up4(d5_with_z8, e4)
        
        if posterior_features is not None:
            pd4 = self.dec.up4(pe5_conditioned, pe4)

        # ===================== Scale 16: 16x16 =====================
        mu_p16, lv_p16 = self.prior_s16(d4)
        if posterior_features is not None:
            mu_q16, lv_q16 = self.post_s16(pd4)
            z16 = mu_q16 + torch.randn_like(mu_q16) * torch.exp(0.5 * lv_q16)
            KLs.append(gaussian_kl_spatial(mu_q16, lv_q16, mu_p16, lv_p16))
            pe4_conditioned = self.pin_s16(pd4, z16)
        else:
            z16 = mu_p16 + torch.randn_like(mu_p16) * torch.exp(0.5 * lv_p16)

        d4_with_z16 = self.inj_s16(d4, z16)
        d3 = self.dec.up3(d4_with_z16, e3)
        
        if posterior_features is not None:
            pd3 = self.dec.up3(pe4_conditioned, pe3)

        # ===================== Scale 32: 32x32 =====================
        mu_p32, lv_p32 = self.prior_s32(d3)
        if posterior_features is not None:
            mu_q32, lv_q32 = self.post_s32(pd3)
            z32 = mu_q32 + torch.randn_like(mu_q32) * torch.exp(0.5 * lv_q32)
            KLs.append(gaussian_kl_spatial(mu_q32, lv_q32, mu_p32, lv_p32))
            pe3_conditioned = self.pin_s32(pd3, z32)
        else:
            z32 = mu_p32 + torch.randn_like(mu_p32) * torch.exp(0.5 * lv_p32)

        d3_with_z32 = self.inj_s32(d3, z32)
        d2 = self.dec.up2(d3_with_z32, e2)
        
        if posterior_features is not None:
            pd2 = self.dec.up2(pe3_conditioned, pe2)

        # ===================== Scale 64: 64x64 =====================
        mu_p64, lv_p64 = self.prior_s64(d2)
        if posterior_features is not None:
            mu_q64, lv_q64 = self.post_s64(pd2)
            z64 = mu_q64 + torch.randn_like(mu_q64) * torch.exp(0.5 * lv_q64)
            KLs.append(gaussian_kl_spatial(mu_q64, lv_q64, mu_p64, lv_p64))
            pe2_conditioned = self.pin_s64(pd2, z64)
        else:
            z64 = mu_p64 + torch.randn_like(mu_p64) * torch.exp(0.5 * lv_p64)

        d2_with_z64 = self.inj_s64(d2, z64)
        d1 = self.dec.up1(d2_with_z64, e1)
        
        if posterior_features is not None:
            pd1 = self.dec.up1(pe2_conditioned, pe1)

        # ===================== Scale 128: 128x128 (Final) =====================
        mu_p128, lv_p128 = self.prior_s128(d1)
        if posterior_features is not None:
            mu_q128, lv_q128 = self.post_s128(pd1)
            z128 = mu_q128 + torch.randn_like(mu_q128) * torch.exp(0.5 * lv_q128)
            KLs.append(gaussian_kl_spatial(mu_q128, lv_q128, mu_p128, lv_p128))
        else:
            z128 = mu_p128 + torch.randn_like(mu_p128) * torch.exp(0.5 * lv_p128)

        # Final injection and prediction
        d1_final = self.inj_s128(d1, z128)
        logits = self.head(d1_final)

        # =================================================================
        # AGGREGATE KL LOSS AND STORE INFO
        # =================================================================
        
        if KLs:
            info["KL_sum"] = torch.stack(KLs, dim=0).sum(0).mean()  # Sum all 8 KL terms
            info["KL_per_scale"] = [kl.mean().item() for kl in KLs]  # For debugging
        else:
            info["KL_sum"] = torch.tensor(0.0, device=x.device)

        # Store all distribution parameters for debugging/analysis
        info.update({
            "mu_p1": mu_p1, "lv_p1": lv_p1, "mu_p2": mu_p2, "lv_p2": lv_p2,
            "mu_p4": mu_p4, "lv_p4": lv_p4, "mu_p8": mu_p8, "lv_p8": lv_p8,
            "mu_p16": mu_p16, "lv_p16": lv_p16, "mu_p32": mu_p32, "lv_p32": lv_p32,
            "mu_p64": mu_p64, "lv_p64": lv_p64, "mu_p128": mu_p128, "lv_p128": lv_p128,
        })

        return logits, info