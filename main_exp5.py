from scipy.interpolate import griddata
from utils.read_data_exp5 import reconstuct_data, Mydataset
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
import os
import time
from torch import optim
from torch.optim import lr_scheduler
import torch.nn as nn
from utils.metrics import metric
from models.PINN_add import PINNs,loss,loss_data #TODO loss代表物理+数据的完整PINNs，loss_data代表只有数据损失
# from models.PINNsformer import EncoderDecoderTransformer
import matplotlib.pyplot as plt
from utils.plots import plots
from models.KAN import KAN
import warnings
warnings.filterwarnings("ignore")
kl_weight = 0.01 # 贝叶斯神经网络的损失权重
#TODO 早停机制
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save([model[i].state_dict() for i in range(len(model))], path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss



def select_optimizer(parms,learning_rate):
    model_optim = optim.Adam(parms, lr=learning_rate)
    return model_optim

def select_criterion():
    criterion = nn.MSELoss()
    return criterion

def reverse_(f_scaler,l_scaler,features,labels):
    if features is not None:
        for i in range(features.shape[1]):
            features[:,i] = features[:,i] * f_scaler[i]
    if labels is not None:
        for i in range(labels.shape[1]):
            labels[:,i] = labels[:,i] * l_scaler[i]
    return features,labels


def vali(model, vali_loader, criterion, device, use_amp=False, f_scaler=None,l_scaler=None):
    total_loss = []
    x = []
    all_preds = []
    all_trues = []

    for i in range(len(model)):
        model[i].eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # GPU
            # batch_x = torch.tensor(batch_x, requires_grad=True)
            # batch_y = torch.tensor(batch_y, requires_grad=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = []
                    for i in range(len(model)):
                        outputs.append(model[i](batch_x))
                    outputs = torch.cat(outputs, dim=-1)

            else:
                outputs = []
                for i in range(len(model)):
                    outputs.append(model[i](batch_x))
                outputs = torch.cat(outputs, dim=-1)
            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()
            # loss = model.loss(outputs, batch_y, batch_x, criterion)
            loss = criterion(pred, true.reshape(pred.shape[0],pred.shape[1]))
            total_loss.append(loss.item())

            x.append(batch_x.detach().cpu().numpy())
            all_preds.append(pred.numpy())
            all_trues.append(true.numpy())

    # 计算平均损失
    average_loss = np.average(total_loss)

    # 拼接值
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)

    x = np.concatenate(x)
    # 计算 误差指标
    mae, mse, rmse, mape, _, _, _, R2, rrmse = metric(all_preds[:, ], all_trues[:, ])
    _, all_preds = reverse_(f_scaler, l_scaler, None, all_preds)
    x, all_trues = reverse_(f_scaler, l_scaler, x, all_trues)
    # Print metrics
    print(f"Validation Loss: {average_loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, "
          f"MAPE: {mape:.4f}, R2: {R2:.4f}, RRMSE: {rrmse:.4f}")

    for i in range(len(model)):
        model[i].train()
    return average_loss, all_preds, all_trues, x, mae, mse, rmse, mape, R2, rrmse


def train(train_loader,vali_loader,checkpoints_path,epochs,lr,model,use_amp=False,device=None,f_scaler=None,l_scaler=None):
    all_preds, all_trues, x = [],[],[]
    path = os.path.join(checkpoints_path)
    if not os.path.exists(path):
        os.makedirs(path)
    train_losses,vali_losses,mse_,mae_,rmse_,r2_, rrmse_ = [],[],[],[],[],[],[]
    time_now = time.time()

    early_stopping = EarlyStopping(patience=100, verbose=True)
    model_optim = []
    parmss = lambda model: [x for x in model.parameters() if x.requires_grad]
    for model_ in model:
        model_optim.append(select_optimizer(parmss(model_),lr))
    criterion = select_criterion()

    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
    train_steps = len(train_loader)
    # scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
    #                                     steps_per_epoch=train_steps,
    #                                     epochs=epochs,
    #                                     max_lr=lr)
    best_mae = 10e8
    for epoch in range(epochs):
        iter_count = 0
        train_loss = []
        for i in range(len(model)):
            model[i].train()
        epoch_time = time.time()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            iter_count += 1
            for i in range(len(model_optim)):
                model_optim[i].zero_grad()
            batch_x = batch_x.float().to(device)

            batch_y = batch_y.float().to(device)

          # encoder - decoder
            if use_amp:
                with torch.cuda.amp.autocast():
                    batch_x = torch.tensor(batch_x, requires_grad=True).float().cuda()
                    outputs = []
                    for i in range(len(model)):
                        outputs.append(model[i](batch_x))
                    outputs = torch.cat(outputs,dim=-1)

                    # TODO 修改为复合的损失函数，包括物理约束、数据损失和边界
                    loss_ = loss(outputs, batch_y, batch_x, criterion,model[0].k1,model[0].k2)
                    # loss = criterion(outputs, batch_y[:,-1])
                    train_loss.append(loss_.item())
            else:
                # batch_x = torch.tensor(batch_x, requires_grad=True).float().cuda()
                # batch_x = torch.tensor(batch_x, requires_grad=True)
                # batch_y = torch.tensor(batch_y, requires_grad=True)
                batch_x = batch_x.clone().detach().requires_grad_(True)
                outputs = []
                for i in range(len(model)):
                    outputs.append(model[i](batch_x))
                outputs = torch.cat(outputs, dim=-1)
                # TODO 修改为复合的损失函数，包括物理约束、数据损失和边界
                loss_ = loss(outputs, batch_y, batch_x, criterion,model[0].k1,model[0].k2)
                train_loss.append(loss_.item())

            if (i + 1) % 100 == 0:
                print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss_.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

            if use_amp:
                scaler.scale(loss).backward(retain_graph=True)
                [scaler.step(model_optim[i]) for i in range(len(model_optim))]
                scaler.update()
            else:
                loss_.backward(retain_graph=True)
                for i in range(len(model_optim)):
                    model_optim[i].step()


        print("Epoch: {} cost time: {}".format(epoch + 1, (time.time() - epoch_time)))
        train_loss = np.average(train_loss)
        vali_loss,all_preds,all_trues,x, mae, mse, rmse, mape, R2, rrmse = vali(model,vali_loader, criterion, device,f_scaler = f_scaler,l_scaler = l_scaler)
        train_losses.append(train_loss)
        vali_losses.append(vali_loss)
        mse_.append(mse)
        mae_.append(mae)
        rmse_.append(rmse)
        r2_.append(R2)
        rrmse_.append(rrmse)

        if best_mae <= mae:
            pass
        else:
            best_mae = mae
            best_checkpoint = r'savepoints\105best_checkpoint.pth'
            torch.save([model[i].state_dict() for i in range(len(model))], best_checkpoint)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            print("Early stopping")
            break
        if (epoch % 1 == 0) or (epoch == (epochs-1)):
            plot_fig4_three_separate(
                x=x,
                all_trues=all_trues,
                all_preds=all_preds,
                epoch=epoch,
                max_points=8000
            )
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            #
            # ax.scatter(x[:, 1], x[:, 0], all_preds[:, 0],
            #            label="Prediction", alpha=0.25, color="r",
            #            s=2, depthshade=False)
            #
            # ax.scatter(x[:, 1], x[:, 0], all_trues[:, 0],
            #            label="Ground truth", alpha=0.85,
            #            facecolors="none", edgecolors="b",
            #            linewidths=0.25, s=6, depthshade=False)
            #
            # ax.set_xlabel('x',fontsize=15)
            # ax.set_ylabel('t',fontsize=15)
            # ax.set_zlabel('P',fontsize=15)
            # ax.tick_params(axis='x', labelsize=15)
            # ax.tick_params(axis='y', labelsize=15)
            # ax.tick_params(axis='z', labelsize=15)
            # # ax.set_title('')
            # ax.legend()
            #
            # plt.show()


    plots(np.arange(epochs),train_losses,vali_losses,mse_,mae_,rmse_,r2_)
    last_model_path = path + '/' + 'checkpoint.pth'
    pth = torch.load(last_model_path)
    for i in range(len(model)):
        model[i].load_state_dict(pth[i])
    return model

def test(test_loader,model,path,use_amp=False,f_scaler=None,l_scaler=None):
    pth = torch.load(path)
    for i in range(len(model)):
        model[i].load_state_dict(pth[i])

    x = []
    all_preds = []
    all_trues = []
    start = time.time()
    for i in range(len(model)):
        model[i].eval()
    with torch.no_grad():
        for i, (batch_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)  # GPU

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = []
                    for i in range(len(model)):
                        outputs.append(model[i](batch_x))
                    outputs = torch.cat(outputs, dim=-1)

            else:
                outputs = []
                for i in range(len(model)):
                    outputs.append(model[i](batch_x))
                outputs = torch.cat(outputs, dim=-1)

            pred = outputs.detach().cpu()
            true = batch_y.detach().cpu()

            x.append(batch_x.detach().cpu().numpy())
            all_preds.append(pred.numpy())
            all_trues.append(true.numpy())
    print(f"测试用时：{(time.time()-start)}")
    # 拼接值
    all_preds = np.concatenate(all_preds)
    all_trues = np.concatenate(all_trues)
    x = np.concatenate(x)
    _, all_trues = reverse_(f_scaler,l_scaler, None,all_trues)
    x, all_preds = reverse_(f_scaler, l_scaler, x, all_preds)
    # x = f_scaler.inverse_transform(x)
    # all_preds = l_scaler.inverse_transform(all_preds)
    # all_trues = l_scaler.inverse_transform(all_trues)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x[:, 0], all_trues[:, 0], marker='o', label='true', alpha=0.5, color='c',
            linewidth=5)
    ax.plot(x[:, 0], all_preds[:, 0], marker='*', label='pred', alpha=0.5, color='r',
            linewidth=5)
    ax.set_xlabel('t', fontsize=18)
    ax.set_ylabel('p', fontsize=18)
    ax.legend(fontsize=18)
    plt.savefig('time-p.jpg', dpi=300)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)

    # TODO 预测流量的曲线
    W_true = all_trues[:, 5] * all_trues[:, 3] * all_trues[:, 1] * x[:, 2] + \
             (1 - all_trues[:, 5]) * all_trues[:, 4] * all_trues[:, 2] * x[:, 2]

    W_pred = all_preds[:, 5] * all_preds[:, 3] * all_preds[:, 1] * x[:, 2] + \
             (1 - all_preds[:, 5]) * all_preds[:, 4] * all_preds[:, 2] * x[:, 2]

    ax2.plot(x[:, 0], W_true, marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax2.plot(x[:, 0], W_pred, marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax2.set_xlabel('t', fontsize=18)
    ax2.set_ylabel('flow', fontsize=18)
    ax2.tick_params(axis='x', labelsize=18)
    ax2.tick_params(axis='y', labelsize=18)
    ax2.legend(fontsize=18)
    plt.savefig('time-Flow.jpg', dpi=300)

    # TODO 预测流量的曲线(真实)
    fig10 = plt.figure()
    ax10 = fig10.add_subplot(111)
    W_true1 = all_trues[:, 5] * all_trues[:, 3] * all_trues[:, 1] * x[:, 2] + \
              (1 - all_trues[:, 5]) * all_trues[:, 4] * all_trues[:, 2] * x[:, 2]

    W_pred1 = all_preds[:, 5] * all_preds[:, 3] * all_trues[:, 1] * x[:, 2] + \
              (1 - all_preds[:, 5]) * all_preds[:, 4] * all_trues[:, 2] * x[:, 2]

    ax10.plot(x[:, 0], W_true1, marker='o', label='true', alpha=0.5, color='c',
              linewidth=5)
    ax10.plot(x[:, 0], W_pred1, marker='*', label='pred', alpha=0.5, color='r',
              linewidth=5)
    ax10.set_xlabel('t', fontsize=18)
    ax10.set_ylabel('flow', fontsize=18)
    ax10.tick_params(axis='x', labelsize=18)
    ax10.tick_params(axis='y', labelsize=18)
    ax10.legend(fontsize=18)
    plt.savefig('time-Flow_true_velfg.jpg', dpi=300)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)

    ax3.plot(x[:, 0], all_trues[:, 8], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax3.plot(x[:, 0], all_preds[:, 8], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax3.set_xlabel('t', fontsize=18)
    ax3.set_ylabel('ug', fontsize=18)
    ax3.tick_params(axis='x', labelsize=18)
    ax3.tick_params(axis='y', labelsize=18)
    ax3.legend(fontsize=18)
    plt.savefig('time-ug.jpg', dpi=300)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)

    ax4.plot(x[:, 0], all_trues[:, 9], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax4.plot(x[:, 0], all_preds[:, 9], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax4.set_xlabel('t', fontsize=18)
    ax4.set_ylabel('uf', fontsize=18)
    ax4.tick_params(axis='x', labelsize=18)
    ax4.tick_params(axis='y', labelsize=18)
    ax4.legend(fontsize=18)
    plt.savefig('time-uf.jpg', dpi=300)

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(x[:, 0], all_trues[:, 2], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax5.plot(x[:, 0], all_preds[:, 2], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax5.set_xlabel('t', fontsize=18)
    ax5.set_ylabel('velf', fontsize=18)
    ax5.tick_params(axis='x', labelsize=18)
    ax5.tick_params(axis='y', labelsize=18)
    ax5.legend(fontsize=18)
    plt.savefig('time-velf.jpg', dpi=300)

    fig5 = plt.figure()
    ax5 = fig5.add_subplot(111)

    ax5.plot(x[:, 0], all_trues[:, 1], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax5.plot(x[:, 0], all_preds[:, 1], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax5.set_xlabel('t', fontsize=18)
    ax5.set_ylabel('velg', fontsize=18)
    ax5.tick_params(axis='x', labelsize=18)
    ax5.tick_params(axis='y', labelsize=18)
    ax5.legend(fontsize=18)
    plt.savefig('time-velg.jpg', dpi=300)

    fig6 = plt.figure()
    ax6 = fig6.add_subplot(111)

    ax6.plot(x[:, 0], all_trues[:, 5], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax6.plot(x[:, 0], all_preds[:, 5], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax6.set_xlabel('t', fontsize=18)
    ax6.set_ylabel('voidg', fontsize=18)
    ax6.tick_params(axis='x', labelsize=18)
    ax6.tick_params(axis='y', labelsize=18)
    ax6.legend(fontsize=18)
    plt.savefig('time-voidg.jpg', dpi=300)

    fig7 = plt.figure()
    ax7 = fig7.add_subplot(111)

    ax7.plot(x[:, 0], all_trues[:, 3], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax7.plot(x[:, 0], all_preds[:, 3], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax7.set_xlabel('t', fontsize=18)
    ax7.set_ylabel('rhog', fontsize=18)
    ax7.tick_params(axis='x', labelsize=18)
    ax7.tick_params(axis='y', labelsize=18)
    ax7.legend(fontsize=18)
    plt.savefig('time-rhog.jpg', dpi=300)

    fig8 = plt.figure()
    ax8 = fig8.add_subplot(111)

    ax8.plot(x[:, 0], all_trues[:, 4], marker='o', label='true', alpha=0.5, color='c',
             linewidth=5)
    ax8.plot(x[:, 0], all_preds[:, 4], marker='*', label='pred', alpha=0.5, color='r',
             linewidth=5)
    ax8.set_xlabel('t', fontsize=18)
    ax8.set_ylabel('rhof', fontsize=18)
    ax8.tick_params(axis='x', labelsize=18)
    ax8.tick_params(axis='y', labelsize=18)
    ax8.legend(fontsize=18)
    plt.savefig('time-rhof.jpg', dpi=300)

    # fig5 = plt.figure(figsize=(12, 8))
    # ax5 = fig5.add_subplot(111)
    #
    # ax5.plot(x[:, 0], all_trues[:, 10], marker='o', label='真实值', alpha=0.5, color='c',
    #          linewidth=5)
    # ax5.plot(x[:, 0], all_preds[:, 10], marker='*', label='预测值', alpha=0.5, color='r',
    #          linewidth=5)
    # ax5.set_xlabel('时间', fontsize=18)
    # ax5.set_ylabel('壁面温度', fontsize=18)
    # ax5.legend(fontsize=18)
    # plt.savefig('time-httmp.jpg', dpi=300)

    pd.DataFrame(
        np.concatenate((x[:, 0].reshape(-1, 1),
            # all_trues[:, 0].reshape(-1, 1), all_preds[:, 0].reshape(-1, 1),
            W_true.reshape(-1, 1), W_pred.reshape(-1, 1)),
            # all_trues[:, 8].reshape(-1, 1), all_preds[:, 8].reshape(-1, 1),
            # all_trues[:, 9].reshape(-1, 1), all_preds[:, 9].reshape(-1, 1)),
            # all_trues[:, 8].reshape(-1, 1), all_preds[:, 8].reshape(-1, 1)),
            axis=1), columns=['时间',
                              # '压力-真实值','压力-预测值',
                              '流量—真实值', '流量—预测值',
                              # '汽相内能—真实值', '汽相内能—预测值',
                              # '液相内能—真实值', '液相内能—预测值',
                              # '壁面温度—真实值', '壁面温度—预测值',
                              ]
    ).to_excel(save_excels)
    plt.show()


    return all_preds, all_trues, x
def to_numpy(a):
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return np.asarray(a)

def downsample_points(x_axis, t_axis, z_axis, max_points=8000, seed=2026):
    n = len(z_axis)
    if n <= max_points:
        return x_axis, t_axis, z_axis

    rng = np.random.default_rng(seed)

    # 保留 LOCA 突变附近更多点，避免关键区域被随机采样掉
    transient_mask = (t_axis >= 960) & (t_axis <= 1040)
    transient_idx = np.where(transient_mask)[0]
    normal_idx = np.where(~transient_mask)[0]

    n_transient = min(len(transient_idx), int(max_points * 0.45))
    n_normal = max_points - n_transient

    keep_transient = rng.choice(transient_idx, size=n_transient, replace=False) if len(transient_idx) > n_transient else transient_idx
    keep_normal = rng.choice(normal_idx, size=n_normal, replace=False) if len(normal_idx) > n_normal else normal_idx

    keep_idx = np.concatenate([keep_transient, keep_normal])
    rng.shuffle(keep_idx)

    return x_axis[keep_idx], t_axis[keep_idx], z_axis[keep_idx]

def plot_3d_scatter_single(
    x_axis,
    t_axis,
    z_axis,
    title,
    color,
    save_path,
    zlim,
    max_points=8000
):
    x_plot, t_plot, z_plot = downsample_points(
        x_axis, t_axis, z_axis, max_points=max_points
    )

    fig = plt.figure(figsize=(5.2, 4.6), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        x_plot,
        t_plot,
        z_plot,
        c=color,
        s=2,
        alpha=0.9,
        linewidths=0,
        depthshade=False,
        rasterized=True
    )

    ax.set_xlabel("x", fontsize=16, labelpad=3)
    ax.set_ylabel("t", fontsize=16, labelpad=3)
    ax.set_zlabel("P", fontsize=16, labelpad=3)
    # ax.set_title(title, fontsize=13)

    ax.set_xlim(np.min(x_axis), np.max(x_axis))
    ax.set_ylim(np.min(t_axis), np.max(t_axis))
    ax.set_zlim(zlim)

    ax.view_init(elev=22, azim=-55)
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", labelsize=13)
    ax.tick_params(axis="z", labelsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def make_grid(x_axis, t_axis, value, nx=240, nt=240):
    grid_x = np.linspace(np.min(x_axis), np.max(x_axis), nx)
    grid_t = np.linspace(np.min(t_axis), np.max(t_axis), nt)
    X, T = np.meshgrid(grid_x, grid_t)

    Z = griddata(
        points=np.column_stack([x_axis, t_axis]),
        values=value,
        xi=(X, T),
        method="linear"
    )

    Z_nearest = griddata(
        points=np.column_stack([x_axis, t_axis]),
        values=value,
        xi=(X, T),
        method="nearest"
    )

    Z[np.isnan(Z)] = Z_nearest[np.isnan(Z)]
    return X, T, Z


def plot_error_map(x_axis, t_axis, true_value, pred_value, save_path):
    X, T, Z_true = make_grid(x_axis, t_axis, true_value)
    _, _, Z_pred = make_grid(x_axis, t_axis, pred_value)

    Z_err = np.abs(Z_pred - Z_true)

    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=300)

    im = ax.pcolormesh(X, T, Z_err, shading="auto", cmap="magma")

    ax.set_xlabel("x", fontsize=16)
    ax.set_ylabel("t", fontsize=16)
    # ax.set_title("Absolute error", fontsize=13)
    ax.tick_params(labelsize=13)

    cbar = fig.colorbar(im, ax=ax, shrink=0.86, pad=0.03)
    cbar.set_label("|Error|", fontsize=13)
    cbar.ax.tick_params(labelsize=13)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

def plot_fig4_three_separate(x, all_trues, all_preds, epoch, max_points=8000):
    x_np = to_numpy(x)
    true_np = to_numpy(all_trues)[:, 0]
    pred_np = to_numpy(all_preds)[:, 0]

    x_axis = x_np[:, 1]
    t_axis = x_np[:, 0]

    zlim = (
        min(np.min(true_np), np.min(pred_np)),
        max(np.max(true_np), np.max(pred_np))
    )

    plot_3d_scatter_single(
        x_axis=x_axis,
        t_axis=t_axis,
        z_axis=true_np,
        title=f"Ground truth at epoch {epoch}",
        color="royalblue",
        save_path=f"fig4_epoch_{epoch}_ground_truth.png",
        zlim=zlim,
        max_points=max_points
    )

    plot_3d_scatter_single(
        x_axis=x_axis,
        t_axis=t_axis,
        z_axis=pred_np,
        title=f"Prediction at epoch {epoch}",
        color="crimson",
        save_path=f"fig4_epoch_{epoch}_prediction.png",
        zlim=zlim,
        max_points=max_points
    )

    plot_error_map(
        x_axis=x_axis,
        t_axis=t_axis,
        true_value=true_np,
        pred_value=pred_np,
        save_path=f"fig4_epoch_{epoch}_absolute_error.png"
    )
if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser('算例5-物理+数据')
    args.add_argument('--file_path', default='data\算例5_New/105段', help='input file path')
    args.add_argument('--save_path', default='results\算例5', help='output file path')
    args.add_argument('--var_list', type=list,
                      default=['p', 'velgj', 'velfj', 'rhog', 'rhof', 'voidg', 'fwalgj', 'fwalfj', 'ug', 'uf'],
                      help='predict params')
    args.add_argument('--checkpoints_path', default='savepoints', help='save checkpoints')
    args.add_argument('--epochs', default=100, help='epochs to train')
    args.add_argument('--lr', default=0.001, help='learning rate')
    args.add_argument('--train_pipelines', default='105', type=str, choices=['230', '105', '305', '230', '344', '340'])
    args.add_argument('--testpipe', default=['10520'], type=list, help='test pipe')
    args = args.parse_args()

    file_path = args.file_path
    var_list = args.var_list
    checkpoints_path = args.checkpoints_path
    save_excels = os.path.join(args.save_path, file_path.split('/')[-1] + 'PD.xlsx')
    epochs = args.epochs
    lr = args.lr
    #重力加速度设置
    b_ = {
        '415':-9.81,
        '105':0,
        '305':0,
        '230':-9.81,
        '344':0,
        '340':0,
    }
    pipelines = {
        '415':["41501", "41502", "41503", "41504", "41505", "41506", "41507", "41508", "41509", "41510", "41511", "41512", "42001", "42002"],
        '105':["10501"] + [f'105{i:02}' for i in range(2,41)], #流量
        '305':['30501']  + [f'305{i:02}' for i in range(2,41)], #流量
        # '175':["15001", "17501", "17502", "18001"], #流量
        '230':["23001", "23002", "23003", "23004", "23005", "23006", "23007", "23008", "23009", "23010", "23011", "23012"], # 230壁温、uf(液相) ug(汽相)内能；
        '344':["34401"] + [f'344{i:02}' for i in range(2, 41)], #流量
        '340': ["34001"] + [f'340{i:02}' for i in range(2, 41)],  # 流量
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = [PINNs([4,128,128,32,6],0,lossfunc=select_criterion(),if_exp7=False),
             PINNs([4,128,128,32,2],0,lossfunc=select_criterion(),if_exp7=False),
             PINNs([4,128,128,32,2],0,lossfunc=select_criterion(),if_exp7=False)]

    # model = [
    #     KAN([4, 128, 128, 32, 6]),
    #     KAN([4, 128, 128, 32, 2]),
    #     KAN([4, 128, 128, 32, 2])
    # ]
    # model = PINNsTransformer([2,128,128,1],0,lossfunc=select_criterion(),attention_embed_dim=128)
    # model = EncoderDecoderTransformer(input_dim=2, embed_dim=128, num_heads=4,lossfunc=select_criterion())
    model = [mo.to(device) for mo in model]
    # tabel,test_table = reconstuct_data(var_list,file_path,index_time_test=[4e-2,40e-2])
    tabel,test_table = reconstuct_data(var_list,file_path,pipeline=pipelines[args.train_pipelines], index__pipeline=args.testpipe)
    from sklearn.preprocessing import MinMaxScaler
    feature = np.array(tabel)[:,:4]
    label = np.array(tabel)[:,4:]

    # TODO 添加最大值的标记,进行无量纲化
    max_feature_values = []
    max_label_values = []
    for i in range(feature.shape[1]):
        max_feature_values.append(feature[:,i].max())
        if np.max(feature[:,i]) == 0:
           pass
        else:
            feature[:,i] = feature[:,i]/(feature[:,i].max())
    for i in range(label.shape[1]):
        max_label_values.append(label[:,i].max())
        if np.sum(label[:,i]) == 0:
           pass
        else:
            label[:, i] = label[:, i] / (label[:, i].max())

    indices = np.random.permutation(len(feature))

    # 计算训练集的大小
    train_size = int(0.8 * len(feature))

    # 选择前 train_size 个索引作为训练集
    train_indices = indices[:train_size]

    # 选择剩下的索引作为验证集
    val_indices = indices[train_size:]

    # 使用选择的索引创建训练集和验证集
    Dataset_train = Mydataset(feature[train_indices], label[train_indices])
    Train_loader = DataLoader(Dataset_train, batch_size=32, shuffle=True, num_workers=0)

    Dataset_val = Mydataset(feature[val_indices], label[val_indices])
    Val_loader = DataLoader(Dataset_val, batch_size=32, shuffle=True, num_workers=0)

    # Dataset = Mydataset(feature, label)
    # loader = DataLoader(Dataset, batch_size=32, shuffle=True, num_workers=0)
    feature_t = np.array(test_table)[:,:4]
    label_t = np.array(test_table)[:,4:]

    #TODO 测试集去量纲化
    for i in range(feature_t.shape[1]):
        if np.sum(feature_t[:,i]) == 0:
           pass
        else:
            feature_t[:,i] = feature_t[:,i] / (max_feature_values[i])

    for i in range(label_t.shape[1]):
        if np.sum(label_t[:,i]) == 0:
           pass
        else:
            label_t[:,i] = label_t[:,i] / (max_label_values[i])

    Dataset_test = Mydataset(feature_t, label_t)
    test_loader = DataLoader(Dataset_test, batch_size=32, shuffle=False, num_workers=0)

    train(Train_loader, Val_loader, checkpoints_path, epochs, lr, model, use_amp=False, device=device,f_scaler=max_feature_values,l_scaler=max_label_values)

    pth = os.path.join(checkpoints_path,'105best_checkpoint.pth')
    test(test_loader, model, pth, use_amp=False,f_scaler=max_feature_values,l_scaler=max_label_values)


