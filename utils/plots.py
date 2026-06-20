import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def plots(epoch,train_losses,vali_losses,mse_,mae_,rmse_,r2_):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(epoch, train_losses, marker='o', label='训练集损失', alpha=0.5, color='c',
            linewidth=5)
    ax.plot(epoch, vali_losses, marker='*', linestyle='--',label='验证集损失', alpha=0.5, color='r',
            linewidth=5)
    ax.set_xlabel('训练轮数', fontsize=18)
    ax.set_ylabel('损失值', fontsize=18)
    ax.legend(fontsize=18)
    plt.savefig('loss_exp5.jpg', dpi=300)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot(epoch, mse_, marker='o', label='mse', alpha=0.8, color='g',
            linewidth=5)
    ax1.set_xlabel('训练轮数', fontsize=18)
    ax1.set_ylabel('误差值', fontsize=18)
    ax1.legend(fontsize=18)
    plt.savefig('mse_exp5.jpg', dpi=300)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111)
    ax2.plot(epoch, mae_, marker='o', label='mae', alpha=0.8, color='b',
             linewidth=5)
    ax2.set_xlabel('训练轮数', fontsize=18)
    ax2.set_ylabel('误差值', fontsize=18)
    ax2.legend(fontsize=18)
    plt.savefig('mae_exp5.jpg', dpi=300)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.plot(epoch, rmse_, marker='o', label='rmse', alpha=0.8, color='m',
             linewidth=5)
    ax3.set_xlabel('训练轮数', fontsize=18)
    ax3.set_ylabel('误差值', fontsize=18)
    ax3.legend(fontsize=18)
    plt.savefig('rmse_exp5.jpg', dpi=300)

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.plot(epoch, r2_, marker='o', label='r2', alpha=0.8, color='k',
             linewidth=5)
    ax4.set_xlabel('训练轮数', fontsize=18)
    ax4.set_ylabel('分数值', fontsize=18)
    ax4.legend(fontsize=18)
    plt.savefig('r2_exp5.jpg', dpi=300)

