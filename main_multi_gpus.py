
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.seq_dataset_exp5 import build_dataset, GRAVITY
from utils.metrics import metric
from models.pi_llm.pi_llm import PILLM
from models.pi_llm.physics import PhysicsResidual
# 复用单卡脚本的前向/损失/collocation 注入逻辑，避免重复实现
from main_pi_llm import compute_losses, SEG_PATHS
import warnings
warnings.filterwarnings("ignore")


def is_main():
    return (not dist.is_initialized()) or dist.get_rank() == 0


def log(*a):
    if is_main():
        print(*a, flush=True)


def setup_ddp():

    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world = int(os.environ.get('WORLD_SIZE', 1))
    # Windows 不支持 NCCL，用 gloo；Linux 多卡用 nccl
    backend = 'nccl' if (torch.cuda.is_available() and os.name != 'nt') else 'gloo'
    dist.init_process_group(backend=backend, rank=rank, world_size=world)
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world


@torch.no_grad()
def evaluate(model, loader, physics, device, w):

    model.eval()
    preds, trues = [], []
    for batch in loader:
        with torch.enable_grad():
            _, (pred, true), _ = compute_losses(model, physics, batch, device, w)
        preds.append(pred.cpu().numpy())
        trues.append(true.cpu().numpy())
    model.train()
    preds = np.concatenate(preds) if preds else np.zeros((0, 10))
    trues = np.concatenate(trues) if trues else np.zeros((0, 10))
    # 收集各 rank 的预测到主进程
    if dist.is_initialized() and dist.get_world_size() > 1:
        gp, gt = [None] * dist.get_world_size(), [None] * dist.get_world_size()
        dist.all_gather_object(gp, preds)
        dist.all_gather_object(gt, trues)
        preds = np.concatenate([x for x in gp if len(x)])
        trues = np.concatenate([x for x in gt if len(x)])
    mae, mse, rmse, mape, _, _, _, r2, rrmse = metric(preds, trues)
    return mae, mse, rmse, r2, rrmse


def build_optimizer(model, lr, lr_llm):

    llm_params, other = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (llm_params if 'backbone' in n else other).append(p)
    groups = [{'params': other, 'lr': lr}]
    if llm_params:
        groups.append({'params': llm_params, 'lr': lr_llm})
    return torch.optim.Adam(groups)


def train(model, physics, tr_loader, va_loader, tr_sampler, device, args, w):
    opt = build_optimizer(model, args.lr, args.lr_llm)
    best_mae = 1e18
    best_path = os.path.join(args.ckpt, 'pi_llm_ddp_best.pth')
    if is_main():
        os.makedirs(args.ckpt, exist_ok=True)
    bad = 0

    for epoch in range(args.epochs):
        model.train()
        if tr_sampler is not None:
            tr_sampler.set_epoch(epoch)        # 保证各 epoch shuffle 不同
        run = {'data': 0, 'phy': 0, 'sace': 0, 's2t': 0}
        for i, batch in enumerate(tr_loader):
            opt.zero_grad()
            total, _, parts = compute_losses(model, physics, batch, device, w)
            total.backward()
            opt.step()
            for k in run:
                run[k] += parts[k]
            if (i + 1) % args.log_every == 0:
                log(f"  e{epoch+1} it{i+1} total={total.item():.5f} "
                    f"data={parts['data']:.5f} phy={parts['phy']:.4f} "
                    f"sace={parts['sace']:.3f} s2t={parts['s2t']:.3f}")
        n = max(1, len(tr_loader))
        mae, mse, rmse, r2, rrmse = evaluate(model, va_loader, physics, device, w)
        log(f"Epoch {epoch+1}/{args.epochs} | train data={run['data']/n:.5f} "
            f"phy={run['phy']/n:.4f} sace={run['sace']/n:.3f} s2t={run['s2t']/n:.3f} "
            f"| val MAE={mae:.4f} RMSE={rmse:.4f} R2={r2:.4f} RRMSE={rrmse:.4f}")

        if is_main() and mae < best_mae:
            best_mae, bad = mae, 0
            raw = model.module if hasattr(model, 'module') else model
            torch.save(raw.state_dict(), best_path)
            log(f"  * saved best (MAE={mae:.5f})")
        elif mae >= best_mae:
            bad += 1
            if bad >= args.patience:
                log("Early stopping"); break
    return best_path


def main():
    ap = argparse.ArgumentParser('PI-LLM 多卡训练')
    ap.add_argument('--train_pipelines', default='105',
                    choices=['105', '305', '230', '340', '344'])
    ap.add_argument('--file_path', default=None)
    ap.add_argument('--llm_path', default=None, help='Llama-3.2-3B 本地目录')
    ap.add_argument('--epochs', type=int, default=1000)
    ap.add_argument('--batch_size', type=int, default=1, help='batch size')
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr_llm', type=float, default=1e-4)
    ap.add_argument('--T', type=int, default=96)
    ap.add_argument('--d_model', type=int, default=256)
    ap.add_argument('--n_colloc', type=int, default=2)
    ap.add_argument('--patience', type=int, default=10)
    ap.add_argument('--log_every', type=int, default=50)
    ap.add_argument('--ckpt', default='savepoints')
    ap.add_argument('--lambda_data', type=float, default=1.0)
    ap.add_argument('--lambda_phy', type=float, default=0.01)
    ap.add_argument('--lambda_sace', type=float, default=0.20)
    ap.add_argument('--lambda_s2t', type=float, default=0.10)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()

    rank, local_rank, world = setup_ddp()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.llm_path = None
        args.patience = 99

    seg = args.train_pipelines
    fpath = args.file_path or SEG_PATHS[seg]
    max_subs = 2 if args.smoke else None
    tr, va, _ = build_dataset(fpath, seg, T=args.T, max_subs=max_subs)
    log(f"{seg}: train={len(tr)} val={len(va)} | world={world} device={device}")

    tr_sampler = DistributedSampler(tr, num_replicas=world, rank=rank, shuffle=True)
    va_sampler = DistributedSampler(va, num_replicas=world, rank=rank, shuffle=False)
    tr_loader = DataLoader(tr, batch_size=args.batch_size, sampler=tr_sampler, num_workers=0)
    va_loader = DataLoader(va, batch_size=args.batch_size, sampler=va_sampler, num_workers=0)

    dtype = torch.float16 if (args.llm_path and torch.cuda.is_available()) else torch.float32
    model = PILLM(llm_path=args.llm_path, d_model=args.d_model,
                  max_prompt_len=args.T, dtype=dtype).to(device)
    if world > 1:

        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                    find_unused_parameters=True)
    physics = PhysicsResidual(b_x=GRAVITY[seg], n_colloc=args.n_colloc).to(device)
    w = {'data': args.lambda_data, 'phy': args.lambda_phy,
         'sace': args.lambda_sace, 's2t': args.lambda_s2t}

    best = train(model, physics, tr_loader, va_loader, tr_sampler, device, args, w)
    log(f"{best}")
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
