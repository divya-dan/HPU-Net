# spu_net.py - PAPER-EXACT IMPLEMENTATION
from __future__ import annotations
from typing import Tuple, Dict, Any, Optional
import torch
from torch import nn
import torch.nn.functional as F


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


# ============================================================================
# STANDARD U-NET COMPONENTS (PAPER SPECIFICATION)
# ============================================================================

def standard_conv_block(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    """Standard conv block: 3 × (3x3 Conv + BN + ReLU) - Paper specification"""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=True),
        nn.ReLU(inplace=True),
    )


class StandardDown(nn.Module):
    """Downsampling block: avg pool + standard conv block"""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)
        self.block = standard_conv_block(in_ch, out_ch)
    
    def forward(self, x):
        return self.block(self.pool(x))


class StandardUp(nn.Module):
    """Upsampling block: nearest neighbor + concat + standard conv block"""
    def __init__(self, in_ch_up: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = standard_conv_block(in_ch_up + skip_ch, out_ch)
    
    def forward(self, x, skip):
        x = self.up(x)
        # Handle size mismatches
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = F.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x, skip], dim=1))


def _cap_channels(c: int) -> int:
    """Cap channels at 192 (paper specification)"""
    return min(c, 192)


class StandardUNetEncoder(nn.Module):
    """
    Standard U-Net encoder: 5 scales with 3×3 conv blocks.
    Base=32, doubled after each downsampling, capped at 192.
    """
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        C1 = _cap_channels(base)      # 32
        C2 = _cap_channels(base*2)    # 64  
        C3 = _cap_channels(base*4)    # 128
        C4 = _cap_channels(base*8)    # 192 (capped)
        C5 = _cap_channels(base*16)   # 192 (capped)
        
        self.enc1 = standard_conv_block(in_ch, C1)
        self.down1 = StandardDown(C1, C2)
        self.down2 = StandardDown(C2, C3)
        self.down3 = StandardDown(C3, C4)
        self.down4 = StandardDown(C4, C5)
        
        self.channels = (C1, C2, C3, C4, C5)

    def forward(self, x):
        e1 = self.enc1(x)      # 128x128
        e2 = self.down1(e1)    # 64x64
        e3 = self.down2(e2)    # 32x32
        e4 = self.down3(e3)    # 16x16
        e5 = self.down4(e4)    # 8x8
        return e1, e2, e3, e4, e5


class StandardUNetDecoder(nn.Module):
    """Standard U-Net decoder"""
    def __init__(self, chans: tuple[int,int,int,int,int]):
        super().__init__()
        C1, C2, C3, C4, C5 = chans
        
        self.up4 = StandardUp(in_ch_up=C5, skip_ch=C4, out_ch=C4)  # 8->16
        self.up3 = StandardUp(in_ch_up=C4, skip_ch=C3, out_ch=C3)  # 16->32
        self.up2 = StandardUp(in_ch_up=C3, skip_ch=C2, out_ch=C2)  # 32->64
        self.up1 = StandardUp(in_ch_up=C2, skip_ch=C1, out_ch=C1)  # 64->128
        
        self.out_ch = C1

    def forward(self, e1, e2, e3, e4, e5):
        d4 = self.up4(e5, e4)  # 8x8 -> 16x16
        d3 = self.up3(d4, e3)  # 16x16 -> 32x32  
        d2 = self.up2(d3, e2)  # 32x32 -> 64x64
        d1 = self.up1(d2, e1)  # 64x64 -> 128x128
        return d1


# ============================================================================
# sPUNet COMPONENTS (PAPER SPECIFICATION)
# ============================================================================

class PriorNetwork(nn.Module):
    """
    Separate prior network that mirrors the U-Net encoder.
    Takes image input, produces 6 global latent distribution parameters.
    """
    def __init__(self, in_ch: int, base: int, z_dim: int):
        super().__init__()
        # Mirror the main encoder architecture
        self.encoder = StandardUNetEncoder(in_ch=in_ch, base=base)
        C1, C2, C3, C4, C5 = self.encoder.channels
        
        # Global average pooling + linear layers for latent parameters
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mu_head = nn.Sequential(
            nn.Linear(C5, C5), nn.ReLU(inplace=True),
            nn.Linear(C5, z_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(C5, C5), nn.ReLU(inplace=True),
            nn.Linear(C5, z_dim)
        )

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode through mirrored encoder
        _, _, _, _, e5 = self.encoder(x)  # Only need deepest features
        
        # Global pooling and latent prediction
        h = self.global_pool(e5).flatten(1)  # [B, C5]
        mu = self.mu_head(h)                 # [B, z_dim]
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)  # [B, z_dim]
        
        return mu, logvar


class PosteriorNetwork(nn.Module):
    """
    Separate posterior network that mirrors the U-Net encoder.
    Takes concatenated image+mask input, produces 6 global latent distribution parameters.
    """
    def __init__(self, in_ch: int, base: int, z_dim: int):
        super().__init__()
        # Input is image+mask concatenated: in_ch + 1
        self.encoder = StandardUNetEncoder(in_ch=in_ch + 1, base=base)
        C1, C2, C3, C4, C5 = self.encoder.channels
        
        # Global average pooling + linear layers
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.mu_head = nn.Sequential(
            nn.Linear(C5, C5), nn.ReLU(inplace=True),
            nn.Linear(C5, z_dim)
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(C5, C5), nn.ReLU(inplace=True),
            nn.Linear(C5, z_dim)
        )

    def forward(self, xy) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encode through mirrored encoder
        _, _, _, _, e5 = self.encoder(xy)  # Only need deepest features
        
        # Global pooling and latent prediction
        h = self.global_pool(e5).flatten(1)  # [B, C5]
        mu = self.mu_head(h)                 # [B, z_dim]
        logvar = self.logvar_head(h).clamp(-10.0, 10.0)  # [B, z_dim]
        
        return mu, logvar


class CombinerNetwork(nn.Module):
    """
    Combiner network: 3 final 1×1 convolutions.
    Combines decoder features with broadcast global latent to produce logits.
    """
    def __init__(self, dec_ch: int, z_dim: int):
        super().__init__()
        # Project latent to feature space
        self.proj_z = nn.Sequential(
            nn.Linear(z_dim, dec_ch),
            nn.ReLU(inplace=True)
        )
        
        # 3 final 1×1 convolutions (paper spec)
        self.combiner = nn.Sequential(
            nn.Conv2d(dec_ch + dec_ch, dec_ch, kernel_size=1, bias=False), 
            nn.BatchNorm2d(dec_ch), nn.ReLU(inplace=True),
            nn.Conv2d(dec_ch, dec_ch, kernel_size=1, bias=False), 
            nn.BatchNorm2d(dec_ch), nn.ReLU(inplace=True),  
            nn.Conv2d(dec_ch, 1, kernel_size=1)  # Final logits
        )

    def forward(self, dec_feat: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dec_feat: Decoder features [B, dec_ch, H, W]  
            z: Global latent [B, z_dim]
        Returns:
            logits: [B, 1, H, W]
        """
        B, dec_ch, H, W = dec_feat.shape
        
        # Project and broadcast latent
        z_proj = self.proj_z(z)  # [B, dec_ch]
        z_broadcast = z_proj.view(B, dec_ch, 1, 1).expand(B, dec_ch, H, W)  # [B, dec_ch, H, W]
        
        # Concatenate and combine
        combined = torch.cat([dec_feat, z_broadcast], dim=1)  # [B, 2*dec_ch, H, W]
        logits = self.combiner(combined)  # [B, 1, H, W]
        
        return logits


# ============================================================================
# MAIN sPUNet MODEL (PAPER-EXACT IMPLEMENTATION)
# ============================================================================

class sPUNet(nn.Module):
    """
    Standard Probabilistic U-Net (sPU-Net) - Paper-exact implementation.
    
    Architecture (per paper):
    - 5-scale standard U-Net (3×3 conv blocks, not ResNet)
    - Separate prior network (mirrors encoder)  
    - Separate posterior network (mirrors encoder)
    - 6 global latent variables
    - 3 final 1×1 convolutions in combiner
    
    Training (per paper):
    - Standard ELBO loss (β = 1, fixed)
    - Base channels = 32
    - Learning rate: 0.5×10⁻⁵ → 1×10⁻⁶ in 5 steps
    - Batch size = 32
    - 240k iterations
    """
    def __init__(self, in_ch: int = 1, base: int = 32, z_dim: int = 6):
        super().__init__()
        self.z_dim = z_dim
        
        # Main encoder-decoder (standard U-Net)
        self.encoder = StandardUNetEncoder(in_ch=in_ch, base=base)
        self.decoder = StandardUNetDecoder(self.encoder.channels)
        
        # Separate prior and posterior networks (paper specification)
        self.prior_net = PriorNetwork(in_ch=in_ch, base=base, z_dim=z_dim)
        self.posterior_net = PosteriorNetwork(in_ch=in_ch, base=base, z_dim=z_dim)
        
        # Combiner network (3 final 1×1 convolutions)
        self.combiner = CombinerNetwork(self.decoder.out_ch, z_dim)
        
        # Paper-specified weight initialization
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
            elif isinstance(m, nn.Linear):
                # Orthogonal for linear layers too
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=0.001)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                # Standard BN initialization
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, y_target: Optional[torch.Tensor] = None, 
                sample_posterior: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass for sPU-Net (paper-exact).
        
        Args:
            x: Input image [B,1,H,W]
            y_target: Target segmentation [B,1,H,W] (for posterior sampling)  
            sample_posterior: If True and y_target provided, sample from posterior
            
        Returns:
            logits: Predicted logits [B,1,H,W]
            info: Dict with distribution parameters and KL divergence
        """
        # Encode image through main encoder
        e1, e2, e3, e4, e5 = self.encoder(x)
        
        # Decode to get feature representation
        dec_feat = self.decoder(e1, e2, e3, e4, e5)  # [B, dec_ch, 128, 128]
        
        # Prior distribution p(z|x) from separate prior network
        mu_p, logvar_p = self.prior_net(x)  # [B, z_dim]
        
        mu_q = logvar_q = None
        if (y_target is not None) and sample_posterior:
            # Posterior distribution q(z|x,y) from separate posterior network
            xy = torch.cat([x, y_target], dim=1)  # [B, 2, H, W]
            mu_q, logvar_q = self.posterior_net(xy)  # [B, z_dim]
            
            # Reparameterization trick: sample from posterior
            std_q = torch.exp(0.5 * logvar_q)
            eps = torch.randn_like(std_q)
            z = mu_q + eps * std_q
        else:
            # Sample from prior p(z|x)
            std_p = torch.exp(0.5 * logvar_p)
            eps = torch.randn_like(std_p)
            z = mu_p + eps * std_p

        # Combine decoder features with latent to get logits
        logits = self.combiner(dec_feat, z)  # [B, 1, H, W]

        # Compute KL divergence if both posterior and prior are available
        if (mu_q is not None) and (logvar_q is not None):
            kl = gaussian_kl(mu_q, logvar_q, mu_p, logvar_p)  # [B]
        else:
            kl = torch.zeros(x.size(0), device=x.device)

        info: Dict[str, Any] = {
            "mu_p": mu_p, "logvar_p": logvar_p,
            "mu_q": mu_q, "logvar_q": logvar_q,
            "z": z,
            "kl": kl,  # [B] - per sample KL
        }

        return logits, info