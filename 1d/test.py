import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


################## plot ground truth, predictions and residuals ###############
ntest=100
x_data = np.load('spde_1_f.npy')
y_data = np.load('spde_1_u.npy')
x_grid = np.linspace(0, 1, x_data.shape[-1])
x_grid = x_grid[::5]
y_test = y_data[-ntest:, ::5]
pred_fno=np.load('pred_fno.npy')
pred_spno=np.load('pred_spno.npy')

fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].set_title("ground truth", fontsize=18)
for i in range(100):
    axs[0, 0].plot(x_grid, y_test[i],'k-')
axs[0, 0].set_xlabel("x", fontsize=15)
axs[0, 0].set_ylabel("$u_g$", fontsize=15)
axs[0, 0].grid(True)
axs[0, 1].set_title("FNO prediction", fontsize=18)
for i in range(ntest):
    axs[0, 1].plot(x_grid, pred_fno[i],'k-')
axs[0, 1].set_xlabel("x", fontsize=15)
axs[0, 1].set_ylabel("$u_p$", fontsize=15)
axs[0, 1].grid(True)
axs[1, 0].set_title("ComFNO prediction", fontsize=18)
for i in range(ntest):
    axs[1, 0].plot(x_grid, pred_spno[i],'k-')
axs[1,0].set_xlabel("x", fontsize=15)
axs[1,0].set_ylabel("$u_p$", fontsize=15)
axs[1, 0].grid(True)
residual = np.load('residual_spno_1d.npy')
r_fno = np.load('residual_fno_1d.npy')
for i in range(100):
    if not i:
        axs[1, 1].plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='FNO',linewidth=1)
        axs[1, 1].plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
    else:
        axs[1, 1].plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
        axs[1, 1].plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
axs[1, 1].set_ylim([-0.045, 0.15])
axs[1, 1].grid(True)
axs[1, 1].set_title("Residuals", fontsize=18)
axs[1, 1].set_xlabel("$x$", fontsize=15)
axs[1, 1].set_ylabel('$u_p-u_g$', fontsize=15)
axs[1, 1].legend(loc = 'upper left')
axins = axs[1, 1].inset_axes((0.38, 0.5, 0.52, 0.45))
for i in range(100):
    if not i:
        axins.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='fno',linewidth=1)
        axins.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
    else:
        axins.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
        axins.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
xlim0, xlim1, ylim0, ylim1 = 0, 1, -0.01, 0.01
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0, tx1, tx1, tx0, tx0]
sy = [ty0, ty0, ty1, ty1, ty0]
axs[1, 1].plot(sx, sy, "black")
# # 画两条线
xy = (xlim0, ylim0)
xy2 = (xlim0, ylim1)
con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=axs[1, 1])
axins.add_artist(con)
xy = (xlim1, ylim0)
xy2 = (xlim1, ylim1)
con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=axs[1, 1])
axins.add_artist(con)
fig.tight_layout()
fig.savefig(r'fig.png', dpi=300)



# ############################# plot residuals #################################
# mpl.rcParams['font.family'] = 'serif'
# residual = np.load('residual_spno_1d.npy')
# r_fno = np.load('residual_fno_1d.npy')
# x_grid = np.linspace(0,1,r_fno.shape[-1])
# fig, ax = plt.subplots()
# plt.subplots_adjust(left=0.1, right=0.96, top=0.96, bottom=0.06)
# for i in range(100):
#     if not i:
#         ax.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='FNO',linewidth=1)
#         ax.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
#     else:
#         ax.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
#         ax.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
# ax.set_ylim([-0.045, 0.15])
# ax.grid()
# # ax.set_xlabel("$x$", fontsize=11)
# # ax.set_ylabel('$u_p-u_g$', fontsize=11)
# ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
# ax.set_xticklabels(['0', '', '', '', '1'], fontsize=15)
# ax.set_yticks([-0.025, 0.025, 0.0625, 0.1, 0.15])
# ax.set_yticklabels(['-0.025', '', '', '', '0.15'], fontsize=15)
# ax.legend(loc = 'upper left', fontsize=14)
# axins = ax.inset_axes((0.38, 0.5, 0.52, 0.45))
# # axins = ax.inset_axes((0.18, 0.53, 0.52, 0.45))  # for multiple data distributions case
# # axins = ax.inset_axes((0.28, 0.53, 0.52, 0.45))   # for initial-boundary case
# for i in range(100):
#     if not i:
#         axins.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='fno',linewidth=1)
#         axins.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
#     else:
#         axins.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
#         axins.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
# xlim0, xlim1, ylim0, ylim1 = 0, 1, -0.01, 0.01
# # xlim0, xlim1, ylim0, ylim1 = 0.9, 1, -0.02, 0.02  # for multiple data distributions case
# # xlim0, xlim1, ylim0, ylim1 = -1, 1, -0.02, 0.02   # for turning points case
# # xlim0, xlim1, ylim0, ylim1 = 0, 1, -0.001, 0.001   # for initial-boundary case
# axins.set_xlim(xlim0, xlim1)
# axins.set_ylim(ylim0, ylim1)
# # plot a box
# tx0 = xlim0
# tx1 = xlim1
# ty0 = ylim0
# ty1 = ylim1
# sx = [tx0, tx1, tx1, tx0, tx0]
# sy = [ty0, ty0, ty1, ty1, ty0]
# ax.plot(sx, sy, "black")
#
# # plot two lines
# xy = (xlim0, ylim0)
# xy2 = (xlim0, ylim1)
# con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=ax)
# axins.add_artist(con)
#
# xy = (xlim1, ylim0)
# xy2 = (xlim1, ylim1)
# con = ConnectionPatch(xyA=xy, xyB=xy2, coordsA="data", coordsB="data", axesA=axins, axesB=ax)
# axins.add_artist(con)
# fig.tight_layout()
# axins.tick_params(axis='x', labelbottom=False)
# axins.tick_params(axis='y', labelleft=False)
# fig.savefig(r'ordinary_notp_residual.png', dpi=300)


