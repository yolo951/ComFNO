import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import ConnectionPatch

mpl.rcParams['font.family'] = 'serif'



residual = np.load('residual_spno_multi_eps.npy')
r_fno = np.load('residual_fno_mul_eps.npy')
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
# ax.set_ylim([-0.3, 0.4])
ax.grid()
ax.legend(loc = 'lower left', fontsize=14)
# ax.set_xlabel("$x$", fontsize=12)
# ax.set_ylabel('$u_p-u_g$', fontsize=12)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax.set_xticklabels(['0', '', '', '', '1'], fontsize=15)
ax.set_yticks([-0.15, -0.05, 0.05, 0.15, 0.25])
ax.set_yticklabels(['-0.15', '', '', '', '0.25'], fontsize=15)
axins = ax.inset_axes((0.18, 0.53, 0.52, 0.45))
for i in range(100):
    if not i:
        axins.plot(x_grid, r_fno[i], color='r', linestyle='dashed', zorder=1, label='fno',linewidth=1)
        axins.plot(x_grid, residual[i], color='b', zorder=2, label='comFNO',linewidth=1)
    else:
        axins.plot(x_grid, r_fno[i], color='r', zorder=1, linestyle='dashed',linewidth=1)
        axins.plot(x_grid, residual[i], zorder=2, color='b',linewidth=1)
xlim0, xlim1, ylim0, ylim1 = 0.9, 1, -0.02, 0.02
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
axins.tick_params(axis='x', labelbottom=False)
axins.tick_params(axis='y', labelleft=False)
fig.savefig(r'multiple_residual.png', dpi=300)

