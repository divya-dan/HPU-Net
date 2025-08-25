from __future__ import annotations
import torch
from torch import nn

def conv_block(in_ch: int, out_ch: int, k: int = 3, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=True),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, padding=p, bias=True),
        nn.ReLU(inplace=True),
    )


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.AvgPool2d(2)  # <-- avg pool per paper
        self.block = conv_block(in_ch, out_ch)
    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch_up: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # <-- nearest per paper
        self.conv = conv_block(in_ch_up + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad (unchanged) then concat
        dh = skip.size(2) - x.size(2); dw = skip.size(3) - x.size(3)
        if dh or dw: x = nn.functional.pad(x, [dw//2, dw-dw//2, dh//2, dh-dh//2])
        return self.conv(torch.cat([x, skip], dim=1))


def _cap(c): return min(c, 192)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        C1 = _cap(base)
        C2 = _cap(base*2)
        C3 = _cap(base*4)
        C4 = _cap(base*8)
        C5 = _cap(base*16)
        self.enc1 = conv_block(in_ch, C1)
        self.down1 = Down(C1, C2)
        self.down2 = Down(C2, C3)
        self.down3 = Down(C3, C4)
        self.down4 = Down(C4, C5)
        self.channels = (C1, C2, C3, C4, C5)

    def forward(self, x):
        e1 = self.enc1(x)   # 128
        e2 = self.down1(e1) # 64
        e3 = self.down2(e2) # 32
        e4 = self.down3(e3) # 16
        e5 = self.down4(e4) # 8
        return e1, e2, e3, e4, e5

class UNetDecoder(nn.Module):
    def __init__(self, chans: tuple[int,int,int,int,int]):
        super().__init__()
        C1, C2, C3, C4, C5 = chans
        # stage 4: up from C5, concat skip C4 -> out C4
        self.up4 = Up(in_ch_up=C5, skip_ch=C4, out_ch=C4)
        # stage 3: up from C4, concat skip C3 -> out C3
        self.up3 = Up(in_ch_up=C4, skip_ch=C3, out_ch=C3)
        # stage 2: up from C3, concat skip C2 -> out C2
        self.up2 = Up(in_ch_up=C3, skip_ch=C2, out_ch=C2)
        # stage 1: up from C2, concat skip C1 -> out C1
        self.up1 = Up(in_ch_up=C2, skip_ch=C1, out_ch=C1)
        self.out_ch = C1

    def forward(self, e1, e2, e3, e4, e5):
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return d1

class PreActResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, proj=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.proj = nn.Conv2d(in_ch, out_ch, 1, bias=False) if (proj or in_ch != out_ch) else None

    def forward(self, x):
        out = self.relu(self.bn1(x))
        shortcut = out if self.proj is None else self.proj(out)
        out = self.conv1(out)
        out = self.conv2(self.relu(self.bn2(out)))
        return out + (x if self.proj is None else shortcut)

# --- Residual U-Net backbone (paper spec) ------------------------------------
# 3 pre-activated residual blocks per scale, avg-pool down, nearest up, cap 192.

class ResStack(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 3):
        super().__init__()
        blocks = [PreActResBlock(in_ch, out_ch, proj=True)]
        for _ in range(n_blocks - 1):
            blocks.append(PreActResBlock(out_ch, out_ch))
        self.net = nn.Sequential(*blocks)

    def forward(self, x):
        return self.net(x)

class ResDown(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, n_blocks: int = 3):
        super().__init__()
        self.pool = nn.AvgPool2d(2)                 # avg-pool per paper
        self.stack = ResStack(in_ch, out_ch, n_blocks)
    def forward(self, x):
        return self.stack(self.pool(x))

class ResUp(nn.Module):
    def __init__(self, in_ch_up: int, skip_ch: int, out_ch: int, n_blocks: int = 3):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")  # nearest per paper
        self.stack = ResStack(in_ch_up + skip_ch, out_ch, n_blocks)
    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.size(2) - x.size(2); dw = skip.size(3) - x.size(3)
        if dh or dw:
            x = nn.functional.pad(x, [dw//2, dw - dw//2, dh//2, dh - dh//2])
        x = torch.cat([x, skip], dim=1)
        return self.stack(x)

def _cap(c):  # keep your existing _cap or reuse this – cap channels at 192
    return min(c, 192)

class ResUNetEncoder(nn.Module):
    """
    Residual U-Net encoder with 5 scales.
    Channels: base, 2*base, 4*base, 8*base, 16*base (capped at 192).
    Each scale = 3 pre-act residual blocks.
    """
    def __init__(self, in_ch: int = 1, base: int = 24, n_blocks: int = 3):
        super().__init__()
        C1 = _cap(base)
        C2 = _cap(base*2)
        C3 = _cap(base*4)
        C4 = _cap(base*8)
        C5 = _cap(base*16)

        self.enc1  = ResStack(in_ch, C1, n_blocks)
        self.down1 = ResDown(C1, C2, n_blocks)
        self.down2 = ResDown(C2, C3, n_blocks)
        self.down3 = ResDown(C3, C4, n_blocks)
        self.down4 = ResDown(C4, C5, n_blocks)

        self.channels = (C1, C2, C3, C4, C5)

    def forward(self, x):
        e1 = self.enc1(x)     # 128x128
        e2 = self.down1(e1)   # 64x64
        e3 = self.down2(e2)   # 32x32
        e4 = self.down3(e3)   # 16x16
        e5 = self.down4(e4)   # 8x8
        return e1, e2, e3, e4, e5

class ResUNetDecoder(nn.Module):
    """
    Mirror of ResUNetEncoder: upsample and fuse with skips, 3 residual blocks each.
    """
    def __init__(self, chans: tuple[int, int, int, int, int], n_blocks: int = 3):
        super().__init__()
        C1, C2, C3, C4, C5 = chans
        self.up4 = ResUp(C5, C4, C4, n_blocks)  # 8->16
        self.up3 = ResUp(C4, C3, C3, n_blocks)  # 16->32
        self.up2 = ResUp(C3, C2, C2, n_blocks)  # 32->64
        self.up1 = ResUp(C2, C1, C1, n_blocks)  # 64->128
        self.out_ch = C1

    def forward(self, e1, e2, e3, e4, e5):
        d4 = self.up4(e5, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        return d1