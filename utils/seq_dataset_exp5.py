
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.read_data_exp5 import data_dict, generate_value

VAR_LIST = ['p', 'velgj', 'velfj', 'rhog', 'rhof', 'voidg', 'fwalgj', 'fwalfj', 'ug', 'uf']

INPUT_COLS = ['t', 'x', 'a', 'q']
N_INPUT = len(INPUT_COLS)
N_VAR = len(VAR_LIST)
N_CH = N_INPUT + N_VAR  # 14

PIPELINES = {
    '415': ["41501", "41502", "41503", "41504", "41505", "41506", "41507",
            "41508", "41509", "41510", "41511", "41512", "42001", "42002"],
    '105': ["10501"] + [f'105{i:02}' for i in range(2, 41)],
    '305': ['30501'] + [f'305{i:02}' for i in range(2, 41)],
    '230': ["23001", "23002", "23003", "23004", "23005", "23006",
            "23007", "23008", "23009", "23010", "23011", "23012"],
    '344': ["34401"] + [f'344{i:02}' for i in range(2, 41)],
    '340': ["34001"] + [f'340{i:02}' for i in range(2, 41)],
}

GRAVITY = {'415': -9.81, '105': 0, '305': 0, '230': -9.81, '344': 0, '340': 0}

SEG_DESC = {
    '105': 'intact-loop primary piping (pipeline 105)',
    '305': 'broken-loop branch (pipeline 305)',
    '230': 'reactor core average channel (pipeline 230)',
    '340': 'broken-loop branch (pipeline 340)',
    '344': 'near-break outlet region (pipeline 344)',
    '415': 'pressurizer main volume (pipeline 415)',
}


def build_prompt(seg, sub_name):
    """SaCE 前缀 prompt: 领域先验 + 任务要求 + 数据描述。"""
    domain = ("In a loss-of-coolant accident (LOCA) of a pressurized water "
              "reactor, the coolant is lost through a pipe break, causing rapid "
              "depressurization, abrupt flow reduction and gas-liquid two-phase "
              "transients governed by the two-fluid six-equation model.")
    task = ("Task: given the historical thermal-hydraulic sequence (time, "
            "relative pipe coordinate, flow area, heat flux) at a monitored "
            "location, predict the current pressure, phase velocities, phase "
            "densities, void fraction, wall friction coefficients and phase "
            "internal energies.")
    data = (f"Data: RELAP5-simulated LOCA transient, 1200 s, 0.5 s sampling, "
            f"monitored at {SEG_DESC.get(seg, seg)}, sub-pipe {sub_name}.")
    return domain + " " + task + " " + data


def load_segment_signals(file_path, seg, start_idx=800):

    subs = PIPELINES[seg]
    signals = {}
    for fname in os.listdir(file_path):
        if not fname.endswith('.xlsx') or fname.startswith('~$'):
            continue
        full = os.path.join(file_path, fname)
        excel = pd.read_excel(full, sheet_name=None, header=1)
        for sub in subs:
            if sub not in excel:
                continue
            df = excel[sub].iloc[start_idx:].reset_index(drop=True)
            t = df['time'].to_numpy(dtype=np.float64)
            a = df['avol'].to_numpy(dtype=np.float64)
            q = df['q'].to_numpy(dtype=np.float64)
            x = np.full_like(t, float(generate_value(sub, data_dict)))
            cols = [t, x, a, q] + [df[v].to_numpy(dtype=np.float64) for v in VAR_LIST]
            signals[sub] = np.stack(cols, axis=1)  # [L, 14]
    return signals


def build_windows(signals, seg, T=96):

    samples = []
    for sub, sig in signals.items():
        L = sig.shape[0]
        if L < T:
            continue
        prompt = build_prompt(seg, sub)
        for end in range(T, L):  # end 为窗口后一位，窗口 = [end-T, end)
            win = sig[end - T:end]                 # [T, 14]
            target = sig[end - 1, N_INPUT:]        # [10] 末端物理量
            colloc = sig[end - 1, :N_INPUT]        # [4]  末端 (t,x,a,q)
            samples.append({
                'history': win.astype(np.float32),
                'suffix': win[:, :N_INPUT].astype(np.float32),
                'target': target.astype(np.float32),
                'colloc': colloc.astype(np.float32),
                'prompt': prompt,
                'seg': seg,
            })
    return samples


def fit_normalizer(samples):

    stack = np.concatenate([s['history'] for s in samples], axis=0)  # [N*T, 14]
    scale = np.max(np.abs(stack), axis=0)
    scale[scale == 0] = 1.0
    return scale.astype(np.float32)


class SeqDataset(Dataset):

    def __init__(self, samples, scale):
        self.samples = samples
        self.scale = torch.tensor(scale)              # [14]
        self.s_in = self.scale[:N_INPUT]              # [4]
        self.s_var = self.scale[N_INPUT:]             # [10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        hist = torch.tensor(s['history']) / self.scale
        suffix = torch.tensor(s['suffix']) / self.s_in
        target = torch.tensor(s['target']) / self.s_var
        colloc = torch.tensor(s['colloc']) / self.s_in
        return {
            'history': hist, 'suffix': suffix, 'target': target,
            'colloc': colloc, 'prompt': s['prompt'],
        }


def build_dataset(file_path, seg, T=96, train_ratio=0.8, seed=2026, max_subs=None):
    signals = load_segment_signals(file_path, seg)
    if max_subs is not None:  # smoke 模式：仅取前若干子管道
        signals = {k: signals[k] for k in list(signals)[:max_subs]}
    samples = build_windows(signals, seg, T=T)
    scale = fit_normalizer(samples)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(samples))
    n_train = int(train_ratio * len(samples))
    tr = [samples[i] for i in idx[:n_train]]
    va = [samples[i] for i in idx[n_train:]]
    return SeqDataset(tr, scale), SeqDataset(va, scale), scale
