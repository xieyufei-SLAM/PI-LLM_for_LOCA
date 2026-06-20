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
    def __init__(self,layers:List,dropoutrate,lossfunc,if_exp7,act='tanh'):
        super(PINNs, self).__init__()
        self.module = nn.Sequential(
            *[nn.Linear(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        )
        self.activation = nn.Tanh() if act == 'tanh' else nn.ReLU()
        self.dropout = nn.Dropout(dropoutrate)
        self.lossfunc = lossfunc
        self.if_exp7 = if_exp7
        self.k = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.k = nn.init.normal_(self.k)
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

    def loss(self,outputs, batch_y, batch_x, criterion, alpha=1,beta=1e-3,gamma=1):
        # initboundary_loss = self.initboundary_loss(batch_x,outputs, batch_y)
        pde_loss = self.pde_loss(batch_x,outputs,batch_y)
        data_loss = self.data_loss(outputs,batch_y)

        total_loss = beta*pde_loss + gamma*data_loss
        # total_loss = (1 / (2 * torch.exp(self.log_sigma_data))) * data_loss + \
        #              (1 / (2 * torch.exp(self.log_sigma_pde))) * pde_loss

        return total_loss #最终损失为三项总和

    # 数据损失
    def data_loss(self,pred,true):
        return self.lossfunc(pred,true[:,-1].reshape(pred.shape[0],pred.shape[1]))

    # 物理约束损失
    def pde_loss(self,batch_x,outputs,batch_y):

        alpha_g = outputs.reshape(-1, 1)  # 空泡份额
        if self.if_exp7:
            alpha_f = 1 - alpha_g
            rho_g = 5.904127E-01  # 汽相密度
            rho_f = 9.583876E+02 / 1000  # 液相密度
            v_g = 1  # 汽相速度
            v_f = 1  # 液相速度
        else:
            alpha_f = 1 - alpha_g
            rho_g = batch_y[:,2].reshape(-1, 1) # 汽相密度
            rho_f = batch_y[:,3].reshape(-1, 1)/1000 # 液相密度
            v_g = batch_y[:,0].reshape(-1, 1)   # 汽相速度
            v_f = batch_y[:,1].reshape(-1, 1)  # 液相速度

        # print(f"alpha_g requires_grad: {alpha_g.requires_grad}, shape: {alpha_g.shape}")
        # print(f"rho_g requires_grad: {rho_g.requires_grad}, shape: {rho_g.shape}")

        # Dalpha_grho_gDt = gradients(batch_x, alpha_g*rho_g)[:, 0]
        # Dalpha_grho_fDt = gradients(batch_x, alpha_g*rho_f)[:, 0]
        # Drho_fDt = 0
        # Dalpha_grho_gv_gDt = gradients(batch_x, alpha_g*rho_g*v_g)[:, 1]
        # Drho_fv_fDt = 0
        # Dalpha_grho_fv_fDt = gradients(batch_x, alpha_g*rho_f*v_f)[:, 1]

        # f1 = Dalpha_grho_gDt + Drho_fDt - Dalpha_grho_fDt +  Dalpha_grho_gv_gDt + Drho_fv_fDt - Dalpha_grho_fv_fDt + self.k # 气相方程+液相方程

        #TODO 简化版
        f1 = rho_g * gradients(batch_x, alpha_g)[:, 0] + rho_g * v_g * gradients(batch_x, alpha_g)[:, 1] + self.k # 气相方程

        f2 = rho_f * gradients(batch_x, alpha_g)[:, 0] + rho_f * v_f * gradients(batch_x, alpha_g)[:,
                                                                       1] + self.k  # 液相方程
        true = torch.zeros_like(f2+f1)

        return self.lossfunc(f2+f1,true) #未进行合并均值等操作

    # # 初始条件损失
    # def initboundary_loss(self,batch_x,outputs, batch_y):
    #     return self.lossfunc(pred,true) #未进行合并均值等操作


if __name__ == '__main__':
    Net = PINNs([2,128,128,1],0.1)
    x = torch.randn((32,2))
    print(Net(x).shape)