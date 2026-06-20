
import math
import torch
import torch.nn as nn


def patchify(x, pl=16, stride=8):
    B, T, V = x.shape
    O = math.floor((T - pl) / stride) + 1
    idx = [slice(i * stride, i * stride + pl) for i in range(O)]
    patches = torch.stack([x[:, s, :] for s in idx], dim=1)   # [B,O,pl,V]
    return patches, O


def temporal_derivative(x, dt=0.5):
    return (x[..., 1:, :] - x[..., :-1, :]) / dt


class S2T(nn.Module):


    def __init__(self, in_dim=4, pl=16, stride=8, d_model=256,
                 nhead=4, nlayers=3, mask_ratio=0.25, dt=0.5, dropout=0.1):
        super().__init__()
        self.pl, self.stride, self.mask_ratio, self.dt = pl, stride, mask_ratio, dt
        self.embed = nn.Linear(pl * in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 64, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(enc, nlayers)
        dec = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.decoder = nn.TransformerEncoder(dec, nlayers)
        self.recon = nn.Linear(d_model, pl * in_dim)
        self.in_dim = in_dim

    def _causal_mask(self, O, device):
        """上三角 causal mask：位置 i 只能看到 j<=i (避免未来泄漏)"""
        m = torch.triu(torch.ones(O, O, device=device), diagonal=1).bool()
        return m

    def forward(self, suffix):
        patches, O = patchify(suffix, self.pl, self.stride)   # [B,O,pl,V]
        B = patches.size(0)
        flat = patches.reshape(B, O, -1)                       # [B,O,pl*V]
        tok = self.embed(flat) + self.pos[:, :O]               # [B,O,d]

        # 随机 mask 25% patches
        n_mask = max(1, int(round(self.mask_ratio * O)))
        mask = torch.zeros(B, O, dtype=torch.bool, device=suffix.device)
        for b in range(B):
            sel = torch.randperm(O, device=suffix.device)[:n_mask]
            mask[b, sel] = True

        enc_in = torch.where(mask.unsqueeze(-1), self.mask_token, tok)
        h = self.encoder(enc_in)
        cmask = self._causal_mask(O, suffix.device)
        dec = self.decoder(h, mask=cmask)                      # causal 重构
        rec = self.recon(dec).reshape(B, O, self.pl, self.in_dim)

        # Eq.(42): masked MSE
        mse = ((patches - rec) ** 2).sum(dim=(-1, -2))         # [B,O]
        m = mask.float()
        recon = (mse * m).sum() / m.sum().clamp_min(1.0)
        # Eq.(43): 一阶导数一致性(对所有 patch)
        dgt = temporal_derivative(patches, self.dt)
        drec = temporal_derivative(rec, self.dt)
        deriv = ((dgt - drec) ** 2).mean()
        loss = recon + deriv
        return loss, h.mean(dim=1)                             # hidden 供可选融合
