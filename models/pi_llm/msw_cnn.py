
import torch
import torch.nn as nn
import torch.nn.functional as F

_S = 0.5  # Haar 归一化系数 (1/sqrt(2))**2 用于 2D


def _haar_filters(device, dtype):
    """4 个 2D Haar 分析滤波器 [4,1,2,2]：LL, LH, HL, HH。"""
    ll = torch.tensor([[1, 1], [1, 1]], dtype=dtype)
    lh = torch.tensor([[1, 1], [-1, -1]], dtype=dtype)
    hl = torch.tensor([[1, -1], [1, -1]], dtype=dtype)
    hh = torch.tensor([[1, -1], [-1, 1]], dtype=dtype)
    f = torch.stack([ll, lh, hl, hh]) * _S
    return f.unsqueeze(1).to(device)           # [4,1,2,2]


def dwt2d(x):

    if x.size(2) % 2:
        x = F.pad(x, (0, 0, 0, 1))
    if x.size(3) % 2:
        x = F.pad(x, (0, 1, 0, 0))
    filt = _haar_filters(x.device, x.dtype)
    out = F.conv2d(x, filt, stride=2)          # [B,4,H/2,W/2]
    return out[:, 0:1], out[:, 1:2], out[:, 2:3], out[:, 3:4]


def idwt2d(ll, lh, hl, hh):

    filt = _haar_filters(ll.device, ll.dtype)
    coeffs = torch.cat([ll, lh, hl, hh], dim=1)   # [B,4,h,w]
    return F.conv_transpose2d(coeffs, filt, stride=2)


class MSWCNN(nn.Module):


    def __init__(self, in_ch=14, mid_ch=32, d_out=256, scales=(3, 5, 7)):
        super().__init__()
        self.unify = nn.Conv1d(in_ch, mid_ch, kernel_size=1)   # 1x1, Eq.34
        self.multi = nn.ModuleList([
            nn.Conv1d(mid_ch, mid_ch, kernel_size=k, padding=k // 2)
            for k in scales])                                   # Eq.35
        self.fuse = nn.Sequential(
            nn.Conv2d(2, mid_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1))
        self.out = nn.Linear(mid_ch, d_out)

    def forward(self, history):
        x = history.transpose(1, 2)                 # [B, V, T]
        h0 = self.unify(x)                          # [B, C, T]
        hs = [conv(h0) for conv in self.multi]      # 3 x [B, C, T]
        cat = torch.cat(hs, dim=1)                  # [B, 3C, T]
        img = cat.unsqueeze(1)                      # [B, 1, 3C, T]
        ll, lh, hl, hh = dwt2d(img)
        zero = torch.zeros_like(ll)
        h_low = idwt2d(ll, lh, zero, zero)
        h_high = idwt2d(zero, zero, hl, hh)
        H = min(h_low.size(2), h_high.size(2))
        W = min(h_low.size(3), h_high.size(3))
        branch = torch.cat([h_low[:, :, :H, :W], h_high[:, :, :H, :W]], dim=1)
        feat = self.fuse(branch).flatten(1)         # [B, mid_ch]
        return self.out(feat)                       # [B, d_out]
