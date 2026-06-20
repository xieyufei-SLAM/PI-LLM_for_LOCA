
import torch
import torch.nn as nn


def gradients(x, y, order=1):
    """Dy/Dx，保留计算图以支持高阶/复合反传。"""
    if order == 1:
        return torch.autograd.grad(
            y, x, grad_outputs=torch.ones_like(y),
            create_graph=True, retain_graph=True)[0]
    return gradients(gradients(x, y), x, order=order - 1)


class PhysicsResidual(nn.Module):
    """两流体六方程质量/动量/能量残差损失。

    k1,k2 为可学习源项闭合系数(沿用 PINN_add 的 k1/k2)。
    b_x 为轴向体积力(重力)，按段设置(GRAVITY)。
    """

    def __init__(self, b_x=0.0, criterion=None, n_colloc=4):
        super().__init__()
        self.b_x = b_x
        self.n_colloc = n_colloc          # 物理残差子批大小(控制二阶 autograd 显存)
        self.criterion = criterion or nn.MSELoss()
        self.k1 = nn.Parameter(torch.zeros(1))
        self.k2 = nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.k1)
        nn.init.normal_(self.k2)

    def forward(self, colloc, pred):
        """colloc:[B,4] (需 requires_grad)；pred:[B,10] 无量纲预测。"""
        a_g = pred[:, 5].reshape(-1, 1)        # 空泡份额 alpha_g
        a_f = 1.0 - a_g
        v_g = pred[:, 1].reshape(-1, 1)
        v_f = pred[:, 2].reshape(-1, 1)
        rho_g = pred[:, 3].reshape(-1, 1)
        rho_f = pred[:, 4].reshape(-1, 1)
        p = pred[:, 0].reshape(-1, 1)
        FWG = pred[:, 6].reshape(-1, 1)
        FWF = pred[:, 7].reshape(-1, 1)
        u_g = pred[:, 8].reshape(-1, 1)
        u_f = pred[:, 9].reshape(-1, 1)
        area = colloc[:, 2].reshape(-1, 1)     # 截面积 A
        q = colloc[:, 3].reshape(-1, 1)        # 热流密度
        bx = self.b_x

        # 对窗口末端时空坐标求导：colloc[:,0]=t, colloc[:,1]=x
        def d(y):
            return gradients(colloc, y)

        # --- 质量守恒 Eq.(15)：气相 + 液相 ---
        cont_g = rho_g * d(a_g)[:, 0] + rho_g * v_g * d(a_g)[:, 1] + self.k1
        cont_f = rho_f * d(a_g)[:, 0] + rho_f * v_f * d(a_g)[:, 1] + self.k2
        zero = torch.zeros_like(cont_g + cont_f)
        loss_mass = self.criterion(cont_g + cont_f, zero)

        # --- 动量守恒 Eq.(16) ---
        mom = d(a_f * rho_f * v_f + a_g * rho_g * v_g)[:, 0] \
            + (1.0 / area) * d(area * a_f * rho_f * v_f * v_f
                               + area * a_g * rho_g * v_g * v_g)[:, 1] \
            + d(a_f * p + a_g * p)[:, 1] \
            - a_f * rho_f * bx - a_g * rho_g * bx \
            + a_f * rho_f * FWF + a_g * rho_g * FWG
        loss_mom = self.criterion(mom, torch.zeros_like(mom))

        # --- 能量守恒 Eq.(17) ---
        ene = d(a_f * rho_f * u_f + a_g * rho_g * u_g)[:, 0] \
            + (1.0 / area) * d(area * (a_f * rho_f * u_f * v_f
                                       + a_g * rho_g * u_g * v_g))[:, 1] \
            + d(p)[:, 0] \
            + (1.0 / area) * d(p * area * (a_f * v_f + a_g * v_g))[:, 1] \
            - q.reshape(-1) \
            - (a_g * rho_g * v_g * v_g * u_g).reshape(-1) \
            - (a_f * rho_f * v_f * v_f * u_f).reshape(-1)
        loss_ene = self.criterion(ene, torch.zeros_like(ene))

        loss_phy = loss_mass + loss_mom + loss_ene
        return loss_phy, {'mass': loss_mass.item(),
                          'mom': loss_mom.item(), 'ene': loss_ene.item()}
