# -*- coding: utf-8 -*-
"""PI-LLM 训练脚本 (manuscript Methods / notes §8)。

复合损失 (Eq.25)：Loss = λd·L_data + λphy·L_phy + λSaCE·L_SaCE + λS2T·L_S2T
默认 λd=1, λphy=0.01, λSaCE=0.20, λS2T=0.10。

用法：
  # 冒烟(tiny 骨干, 子集, 少 epoch)
  python main_pi_llm.py --smoke
  # 全量(Llama-3.2-3B + LoRA)
  python main_pi_llm.py --llm_path llm_weights/LLM-Research/Llama-3___2-3B \
      --train_pipelines 105 --epochs 1000
"""
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.seq_dataset_exp5 import build_dataset, GRAVITY, N_INPUT
from utils.metrics import metric
from models.pi_llm.pi_llm import PILLM
from models.pi_llm.physics import PhysicsResidual
import warnings
warnings.filterwarnings("ignore")


def inject_colloc(history, suffix, colloc):
    """把可微 colloc[B,4] 注入 history/suffix 末位时间步的前 4 列，

    使预测对 (t,x,a,q) 可微(供物理残差 autograd)，非 in-place。
    """
    hist = torch.cat([history[:, :-1],
                      torch.cat([colloc, history[:, -1, N_INPUT:]], dim=1).unsqueeze(1)],
                     dim=1)
    suf = torch.cat([suffix[:, :-1], colloc.unsqueeze(1)], dim=1)
    return hist, suf


def compute_losses(model, physics, batch, device, w):
    """前向 + 复合损失。返回 (total, data_pred, parts dict)。"""
    history = batch['history'].to(device)
    suffix = batch['suffix'].to(device)
    target = batch['target'].to(device)
    colloc = batch['colloc'].to(device).clone().requires_grad_(True)
    prompts = batch['prompt']

    hist, suf = inject_colloc(history, suffix, colloc)
    out = model(hist, suf, prompts)
    pred = out['pred']

    mse = nn.functional.mse_loss(pred, target)
    loss_phy, phy_parts = physics(colloc, pred)
    loss_sace = out['loss_sace']
    loss_s2t = out['loss_s2t']
    total = (w['data'] * mse + w['phy'] * loss_phy
             + w['sace'] * loss_sace + w['s2t'] * loss_s2t)
    parts = {'data': mse.item(), 'phy': loss_phy.item(),
             'sace': loss_sace.item(), 's2t': loss_s2t.item()}
    return total, (pred.detach(), target.detach()), parts


@torch.no_grad()
def evaluate(model, loader, physics, device, w):
    model.eval()
    preds, trues, totals = [], [], []
    for batch in loader:
        # 物理残差需梯度 -> 验证时也开 grad（仅前向，不更新）
        with torch.enable_grad():
            total, (pred, true), _ = compute_losses(model, physics, batch, device, w)
        preds.append(pred.cpu().numpy())
        trues.append(true.cpu().numpy())
        totals.append(total.item())
    model.train()
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)
    mae, mse, rmse, mape, _, _, _, r2, rrmse = metric(preds, trues)
    return np.mean(totals), mae, mse, rmse, r2, rrmse


def build_optimizer(model, lr, lr_llm):
    """LoRA/主干参数用更小 lr，其余模块用主 lr(分组)。"""
    llm_params, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (llm_params if 'backbone' in n else other).append(p)
    groups = [{'params': other, 'lr': lr}]
    if llm_params:
        groups.append({'params': llm_params, 'lr': lr_llm})
    return torch.optim.Adam(groups)


def train(model, physics, tr_loader, va_loader, device, args, w):
    opt = build_optimizer(model, args.lr, args.lr_llm)
    best_mae, best_path = 1e18, os.path.join(args.ckpt, 'pi_llm_best.pth')
    os.makedirs(args.ckpt, exist_ok=True)
    patience, bad = args.patience, 0

    for epoch in range(args.epochs):
        model.train()
        run = {'data': 0, 'phy': 0, 'sace': 0, 's2t': 0}
        for i, batch in enumerate(tr_loader):
            opt.zero_grad()
            total, _, parts = compute_losses(model, physics, batch, device, w)
            total.backward()
            opt.step()
            for k in run:
                run[k] += parts[k]
            if (i + 1) % args.log_every == 0:
                print(f"  e{epoch+1} it{i+1} total={total.item():.5f} "
                      f"data={parts['data']:.5f} phy={parts['phy']:.4f} "
                      f"sace={parts['sace']:.3f} s2t={parts['s2t']:.3f}")
        n = len(tr_loader)
        vloss, mae, mse, rmse, r2, rrmse = evaluate(model, va_loader, physics, device, w)
        print(f"Epoch {epoch+1}/{args.epochs} | train data={run['data']/n:.5f} "
              f"phy={run['phy']/n:.4f} sace={run['sace']/n:.3f} s2t={run['s2t']/n:.3f} "
              f"| val loss={vloss:.5f} MAE={mae:.4f} RMSE={rmse:.4f} "
              f"R2={r2:.4f} RRMSE={rrmse:.4f}")
        if mae < best_mae:
            best_mae, bad = mae, 0
            torch.save(model.state_dict(), best_path)
            print(f"  * saved best (MAE={mae:.5f})")
        else:
            bad += 1
            if bad >= patience:
                print("Early stopping"); break
    return best_path


SEG_PATHS = {
    '105': 'data/算例5_New/105段', '305': 'data/算例5_New/305段',
    '340': 'data/算例5_New/340段', '344': 'data/算例5/344段',
    '230': 'data/算例5/230段',
}


def main():
    ap = argparse.ArgumentParser('PI-LLM 训练 (算例5)')
    ap.add_argument('--train_pipelines', default='105',
                    choices=['105', '305', '230', '340', '344'])
    ap.add_argument('--file_path', default=None, help='覆盖段数据路径')
    ap.add_argument('--llm_path', default=None, help='Llama-3.2-3B 本地目录')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--lr_llm', type=float, default=1e-4)
    ap.add_argument('--T', type=int, default=96)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--ckpt', default='savepoints')
    # 复合损失权重 (notes §3)
    ap.add_argument('--lambda_data', type=float, default=1.0)
    ap.add_argument('--lambda_phy', type=float, default=0.01)
    ap.add_argument('--lambda_sace', type=float, default=0.20)
    ap.add_argument('--lambda_s2t', type=float, default=0.10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.llm_path = None
        args.patience = 99
    seg = args.train_pipelines
    fpath = args.file_path or SEG_PATHS[seg]
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    max_subs = 2 if args.smoke else None
    tr, va, scale = build_dataset(fpath, seg, T=args.T, max_subs=max_subs)
    print(f"段 {seg}: train={len(tr)} val={len(va)} | device={device}")
    tr_loader = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)

    dtype = torch.bfloat16 if (args.llm_path and torch.cuda.is_available()) else torch.float32
    model = PILLM(llm_path=args.llm_path, d_model=args.d_model,
                  max_prompt_len=args.T, dtype=dtype).to(device)
    physics = PhysicsResidual(b_x=GRAVITY[seg]).to(device)
    w = {'data': args.lambda_data, 'phy': args.lambda_phy,
         'sace': args.lambda_sace, 's2t': args.lambda_s2t}

    best = train(model, physics, tr_loader, va_loader, device, args, w)
    print(f"完成。最佳权重: {best}")


if __name__ == '__main__':
    main()
