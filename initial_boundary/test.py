import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# residual=np.load('residual_spno_1d.npy')
# r_fno=np.load('residual_fno_1d.npy')
# x_grid=np.linspace(0,1,r_fno.shape[-1])
# fig=plt.figure()
# for i in range(100):
#     if not i:
#         plt.plot(x_grid, r_fno[i],color='r',linestyle='dashed',zorder=1,label='fno')
#         plt.plot(x_grid, residual[i],color='b',zorder=2,label='comfno')
#     else:
#         plt.plot(x_grid, r_fno[i], color='r', zorder=1,linestyle='dashed')
#         plt.plot(x_grid, residual[i], zorder=2,color='b')
# plt.ylim([-0.02, 0.02])
# plt.grid()
# plt.legend()
# plt.show(block=True)


# import matplotlib.pyplot as plt
# import matplotlib
# import matplotlib as mpl
# from matplotlib.patches import ConnectionPatch
# from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# residual=np.load('comFNO/residual_spno_1d.npy')
# x_grid=np.linspace(0,1,residual.shape[-1])
# fig=plt.figure()
# for i in range(100):
#     plt.plot(x_grid, residual[i],color='b',zorder=2)
# plt.grid()
# plt.show(block=True)

mpl.rcParams['font.family'] = 'serif'



residual = np.load('residual_spno_ib.npy')
r_fno = np.load('residual_fno_ib.npy')
x_grid = np.linspace(0,1,r_fno.shape[-1])
fig, ax = plt.subplots()
plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.06)
for i in range(100):
    if not i:
        ax.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='FNO',linewidth=1)
        ax.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
    else:
        ax.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
        ax.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
ax.set_ylim([-0.006, 0.009])
ax.grid()
# ax.set_xlabel("$x$", fontsize=12)
# ax.set_ylabel('$u_p-u_g$', fontsize=12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0', '', '', '', '1'], fontsize=15)
ax.set_yticks([-0.006, -0.003, 0, 0.003, 0.006, 0.008])
ax.set_yticklabels(['-0.006', '', '', '', '', '0.008'], fontsize=15)
ax.legend(loc = 'lower left', fontsize=14)
axins = ax.inset_axes((0.28, 0.53, 0.52, 0.45))
for i in range(100):
    if not i:
        axins.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='fno',linewidth=1)
        axins.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
    else:
        axins.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
        axins.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
xlim0, xlim1, ylim0, ylim1 = 0, 1, -0.001, 0.001
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)
# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0, tx1, tx1, tx0, tx0]
sy = [ty0, ty0, ty1, ty1, ty0]
ax.plot(sx, sy, "black")

# 画两条线
xy = (xlim0, ylim0)
xy2 = (xlim0, ylim1)
con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=ax)
axins.add_artist(con)

xy = (xlim1, ylim0)
xy2 = (xlim1, ylim1)
con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=ax)
axins.add_artist(con)
plt.tight_layout()
axins.tick_params(axis='x', labelbottom=False)
axins.tick_params(axis='y', labelleft=False)
fig.savefig(r'initialboundary_residual.png', dpi=300)





#
# x_test = x_data[:ntrain, :]
# y_test = y_data[:ntrain,:]
# y_test = torch.Tensor(y_test)
# x_test = torch.Tensor(x_test)
# x_test = torch.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# x_grid = np.linspace(0, 1, x_test.shape[1])
# pred = torch.zeros(y_test.shape)
# test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
# index = 0
# test_l2 = 0
# test_mse = 0
# with torch.no_grad():
#     for x, y in test_loader:
#         x, y = x.cuda(), y.cuda()
#
#         out = model(x).view(-1)
#         pred[index] = out
#         mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
#         test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
#         test_mse += mse.item()
#         index += 1
#
#     test_mse /= len(test_loader)
#     test_l2 /= ntrain
#     print('test error: L2 =', test_l2, ', MSE =', test_mse)
# residual = pred - y_test
# fig = plt.figure(figsize=(8, 6), dpi=150)
#
# plt.subplot(2, 2, 1)
# plt.title("ground truth")
# for i in range(ntrain):
#     plt.plot(x_grid, y_test[i])
# plt.xlabel("x")
# plt.ylabel("$u_g$")
# plt.grid()
#
# plt.subplot(2, 2, 2)
# plt.title("prediction")
# for i in range(ntrain):
#     plt.plot(x_grid, pred[i])
# plt.xlabel("x")
# plt.ylabel("$u_p$")
# plt.grid()
#
# plt.subplot(2, 2, 3)
# plt.title("residuals")
# # plt.ylim([-0.01, 0.01])
# for i in range(ntrain):
#     plt.plot(x_grid, residual[i])
# plt.xlabel("x")
# plt.ylabel("$u_g$-$u_p$")
# plt.grid()
#
# plt.subplot(2, 2, 4)
# plt.title("training loss")
# plt.plot(MSE, c='r', label='MSE')
# plt.plot(L2, c='b', label='Relative')
# plt.legend()
# plt.yscale('log')
# plt.xlabel("epoch")
# plt.ylabel("error")
# plt.grid()
#
# plt.tight_layout()
# plt.show(block=True)