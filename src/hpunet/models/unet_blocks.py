from __future__ import annotations
import torch
from torch import nn

def conv_block(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=k, stride=1, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = conv_block(in_ch, out_ch)
    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class Up(nn.Module):
    """
    Upsample + concat with skip, then conv on (in_ch_up + skip_ch) -> out_ch.
    """
    def __init__(self, in_ch_up: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = conv_block(in_ch_up + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        # pad if sizes off by 1
        dh = skip.size(2) - x.size(2)
        dw = skip.size(3) - x.size(3)
        if dh != 0 or dw != 0:
            x = nn.functional.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class UNetEncoder(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        C1, C2, C3, C4, C5 = base, base*2, base*4, base*8, base*16
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
