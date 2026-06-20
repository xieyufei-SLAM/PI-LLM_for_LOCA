
"""SaCE: Suffix-aware Contrastive Embedding

T=96, L=10, Δ=20, γ=0.5,
σ_η²=0.01, τ_c=0.07。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalEncoder(nn.Module):
    """[B, T, 4] -> [B, d]"""

    def __init__(self, in_dim=4, d_model=256, nhead=4, nlayers=2, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, 512, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=d_model * 2,
            dropout=dropout, activation='gelu', batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayers)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, y):
        h = self.proj(y) + self.pos[:, :y.size(1)]
        h = self.encoder(h)
        return self.out(h.mean(dim=1))


def detect_key_moment(suffix):

    t = suffix[:, :, 0]
    a = suffix[:, :, 2]
    q = suffix[:, :, 3]
    dt = (t[:, 1:] - t[:, :-1]).clamp_min(1e-6)
    da = (a[:, 1:] - a[:, :-1]) / dt
    dq = (q[:, 1:] - q[:, :-1]) / dt
    r = torch.sqrt(da ** 2 + dq ** 2)             # [B, T-1]
    j_star = r.argmax(dim=1) + 1                   # 对齐到原序列下标
    return j_star


def select_positive(suffix, j_star, L):
    B, T, C = suffix.shape
    out = suffix.new_zeros(B, L, C)
    for b in range(B):
        end = int(j_star[b]) + 1
        start = max(0, end - L)
        seg = suffix[b, start:end]
        out[b, L - seg.size(0):] = seg
    return out


def sample_negatives(suffix, j_star, L, delta, K):
    """距 j* >= Δ 的窗口作为负样本候选，每条取 K 个，长度 L"""
    B, T, C = suffix.shape
    negs = suffix.new_zeros(B, K, L, C)
    for b in range(B):
        js = int(j_star[b])
        cand = [e for e in range(L, T) if abs((e - 1) - js) >= delta]
        if not cand:
            cand = list(range(L, T))
        sel = torch.randint(0, len(cand), (K,))
        for k in range(K):
            end = cand[int(sel[k])]
            negs[b, k] = suffix[b, end - L:end]
    return negs


def add_hard_negatives(negs, gamma, sigma_eta):
    """时间置乱 + 凸组合 + 高斯噪声，生成 hard negative。

    """
    B, K, L, C = negs.shape
    perm_t = torch.randperm(L, device=negs.device)
    a = negs
    b = negs[:, torch.randperm(K, device=negs.device)]   # 另一负样本
    b = b[:, :, perm_t, :]                                # 时间置乱 Pi(.)
    eta = torch.randn_like(negs) * (sigma_eta ** 0.5)
    hard = gamma * a + (1.0 - gamma) * b + eta
    return torch.cat([negs, hard], dim=1)                 # [B, 2K, L, C]


def info_nce(p, z_pos, z_neg, tau_c=0.07):
    p = F.normalize(p, dim=-1)
    z_pos = F.normalize(z_pos, dim=-1)
    z_neg = F.normalize(z_neg, dim=-1)
    pos = (p * z_pos).sum(-1, keepdim=True) / tau_c             # [B,1]
    neg = torch.einsum('bd,bmd->bm', p, z_neg) / tau_c          # [B,M]
    logits = torch.cat([pos, neg], dim=1)
    labels = torch.zeros(p.size(0), dtype=torch.long, device=p.device)
    return F.cross_entropy(logits, labels)


class SaCE(nn.Module):

    def __init__(self, d_model=256, L=10, delta=20, gamma=0.5,
                 sigma_eta=0.01, tau_c=0.07, n_neg=8):
        super().__init__()
        self.encoder = TemporalEncoder(in_dim=4, d_model=d_model)
        self.L, self.delta, self.gamma = L, delta, gamma
        self.sigma_eta, self.tau_c, self.n_neg = sigma_eta, tau_c, n_neg

    def encode(self, suffix):
        return self.encoder(suffix)

    def forward(self, suffix, prefix):
        j_star = detect_key_moment(suffix)
        y_pos = select_positive(suffix, j_star, self.L)
        y_neg = sample_negatives(suffix, j_star, self.L, self.delta, self.n_neg)
        y_neg = add_hard_negatives(y_neg, self.gamma, self.sigma_eta)
        z_pos = self.encoder(y_pos)
        B, M, L, C = y_neg.shape
        z_neg = self.encoder(y_neg.reshape(B * M, L, C)).reshape(B, M, -1)
        z_full = self.encode(suffix)
        loss = info_nce(prefix, z_pos, z_neg, self.tau_c)
        return loss, z_full
