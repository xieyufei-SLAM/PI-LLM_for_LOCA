import torch
import torch.nn as nn
from typing import List

from models.KAN import KAN

# 求梯度以及阶数
def gradients(x, y, order=1):
    """Computer the gradient : Dy/Dx."""
    if order == 1:
        return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y),
                                   create_graph=True, retain_graph=True)[0]
    else:
        return gradients(gradients(x, y), x, order=order-1)

class PINNs(nn.Module):
    def __init__(self,layers:List,dropoutrate,lossfunc,if_exp7,act=None):
        super(PINNs, self).__init__()
        self.module = nn.Sequential(
            *[nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        )
        # self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropoutrate)
        self.lossfunc = lossfunc
        self.if_exp7 = if_exp7
        self.k1 = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.k1 = nn.init.normal_(self.k1)

        self.k2 = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.k2 = nn.init.normal_(self.k2)
        self.activation = nn.ReLU() if act == 'relu' else nn.Tanh()
        # self.log_sigma_data = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        # self.log_sigma_pde = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        #
        # self.log_sigma_data = nn.init.normal_(self.log_sigma_data)
        # self.log_sigma_pde = nn.init.normal_(self.log_sigma_pde)

    def forward(self,x):
        for index,modu in enumerate(self.module):
            x = modu(x)
            if index != len(self.module):
                x = self.dropout(x)
                x = self.activation(x)
        return x
    # TODO： 初始条件和边界条件损失在文中为边界位置的样本， 已经包含在样本中，计算规则一样，只是位置不一样，所以可以直接加入 PDE损失
def loss(outputs, batch_y, batch_x, criterion,k1,k2,alpha=1,beta=1e-2,gamma=1):
    # 数据损失
    def data_loss(pred, true, criterion):
        return criterion(pred, true.reshape(pred.shape[0], pred.shape[1]))

    # 物理约束损失
    def pde_loss(batch_x, outputs, batch_y, criterion, k1, k2):
        alpha_g = outputs[:,0].reshape(-1, 1)  # 空泡份额
        alpha_f = 1 - alpha_g
        rho_g = outputs[:, 3].reshape(-1, 1)  # 汽相密度
        rho_f = outputs[:, 4].reshape(-1, 1)  # 液相密度
        v_g = outputs[:, 1].reshape(-1, 1)  # 汽相速度
        v_f = outputs[:, 2].reshape(-1, 1)  # 液相速度
        p = outputs[:, 0].reshape(-1, 1)  # 压力
        FWF = outputs[:, 7].reshape(-1, 1)
        FWG = outputs[:, 6].reshape(-1, 1)
        u_g = outputs[:, 8].reshape(-1, 1) # 汽相内能
        u_f = outputs[:, 9].reshape(-1, 1) # 液相内能
        B_x = 0 # 取决于管道的方向

        # TODO 连续性方程
        Continuity1 = rho_g * gradients(batch_x, alpha_g)[:, 0] + rho_g * v_g * gradients(batch_x, alpha_g)[:, 1] + k1  # 气相方程
        Continuity2 = rho_f * gradients(batch_x, alpha_g)[:, 0] + rho_f * v_f * gradients(batch_x, alpha_g)[:,1] + k2  # 液相方程
        true = torch.zeros_like(Continuity1 + Continuity2)
        C_equation = criterion(Continuity1 + Continuity2, true)

        # TODO 动量方程
        Momentum = gradients(batch_x, alpha_f*rho_f*v_f + alpha_g*rho_g*v_g)[:, 0] + \
                    1/batch_x[:,2]*gradients(batch_x, batch_x[:,2]*alpha_f*rho_f*v_f*v_f + batch_x[:,2]*alpha_g*rho_g*v_g*v_g)[:,1] + \
                    gradients(batch_x, alpha_f * p + alpha_g * p)[:, 1] - alpha_f*rho_f*B_x - alpha_g*rho_g*B_x + \
                    alpha_f * rho_f * FWF + alpha_g * rho_g * FWG # Bx是重力加速度？
        true_M = torch.zeros_like(Momentum)
        M_equation = criterion(Momentum, true_M)

        # TODO 能量方程
        Energy = gradients(batch_x, alpha_f * rho_f * u_f + alpha_g * rho_g * u_g)[:, 0] + \
                   1/batch_x[:,2]*gradients(batch_x,  batch_x[:,2]*(alpha_f * rho_f * u_f * v_f + alpha_g * rho_g * u_g * v_g))[:, 1] + \
                   gradients(batch_x, p)[:,0] + 1/batch_x[:,2]*gradients(batch_x, p*batch_x[:,2]*(alpha_f * v_f + alpha_g * v_g))[:, 1] - batch_x[:, 3] - \
                   alpha_g * rho_g * v_g * v_g * u_g - alpha_f * rho_f * v_f * v_f * u_f
        true_E = torch.zeros_like(Energy)
        E_equation = criterion(Energy, true_E)

        return C_equation + M_equation + E_equation

    dataloss = data_loss(outputs, batch_y, criterion)
    pdeloss = pde_loss(batch_x,outputs,batch_y,criterion,k1,k2)
    total_loss = beta * pdeloss + gamma * dataloss

    return total_loss.mean() #最终损失为三项总和

def loss_data(outputs, batch_y, batch_x, criterion,k1,k2,alpha=1,beta=1e-6,gamma=1):
    # 数据损失
    def data_loss(pred, true, criterion):
        return criterion(pred, true.reshape(pred.shape[0], pred.shape[1]))

    dataloss = data_loss(outputs, batch_y, criterion)

    return dataloss.mean()
