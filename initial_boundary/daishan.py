import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D


def FD_CN_copy(g, gridx ,gridt, epsilon):
    b=lambda x: np.ones_like(x)
    c=lambda x: x
    f=lambda x,t: np.zeros_like(x)


    alpha = 1
    Nx = gridx.shape[0]
    Nt = gridt.shape[0]
    sigma = min(1 / 2, epsilon * np.log(Nx-1) / alpha)
    gridS = np.hstack(
        (np.linspace(0, 1-sigma, int((Nx - 1) / 2) + 1), np.linspace(1-sigma, 1, int((Nx - 1) / 2) + 1)[1:]))
    gS = interpolate.interp1d(gridx, g)(gridS)
    h1= (1 - sigma) / ((Nx - 1) / 2)
    h2 = sigma / ((Nx - 1) / 2)
    dt=1/(Nt-1)
    yS = np.zeros((Nt, Nx))
    U = np.zeros((Nx - 2, Nx - 2))
    U[0, :2] = np.array([2 * epsilon / (h1 ** 2) + c(h1), -epsilon / (h1 ** 2)+b(h1)/2/h1])
    U[-1, -2:] = np.array(
        [-epsilon / (h2 ** 2) - b(1 - h2) /2/h2, 2 * epsilon / (h2 ** 2) + c(1 - h2)])

    Nm = int((Nx - 3) / 2)  # here Nx must be odd
    for i in range(1, Nm):
        x_i = (i + 1) * h1
        p1 = -epsilon / (h1 ** 2) - b(x_i) / 2/h1
        r1 = 2 * epsilon / (h1 ** 2) + c(x_i)
        q1 = -epsilon / (h1 ** 2)+b(x_i) / 2/h1
        U[i, i - 1:i + 2] = np.array([p1, r1, q1])
    x_i = (Nm + 1) * h1
    U[Nm, Nm - 1:Nm + 2] = np.array(
        [-2 * epsilon / (h1 * (h1 + h2)) - b(x_i) / (h1+h2), 2 * epsilon / (h1 * h2) + c(x_i),
         -2 * epsilon / (h2 * (h1 + h2)) + b(x_i) / (h1+h2)])
    for i in range(Nm + 1, Nx - 3):
        x_i = (Nm + 1) * h1 + (i - Nm) * h2
        p2 = -epsilon / (h2 ** 2) - b(x_i) / 2 / h2
        r2 = 2 * epsilon / (h2 ** 2) + c(x_i)
        q2 = -epsilon / (h2 ** 2)+b(x_i) / 2 / h2
        U[i, i - 1:i + 2] = np.array([p2, r2, q2])

    A = np.eye(Nx - 2) + 0.5 * dt * U
    yS[0]=gS
    for j in range(1,Nt):
        B1=0.5*dt*f(gridS,(j+1)*dt)+0.5*dt*f(gridS,j*dt)
        B=np.dot(np.eye(Nx-2)-0.5*dt*U,yS[j-1,1:-1])+B1[1:-1]

        yS[j, 0] = 0
        yS[j, 1:-1] = np.linalg.solve(A, B).flatten()
        yS[j, -1] = 0
    y = interpolate.interp1d(gridS, yS)(gridx)
    return y


x_data=np.load('spde_ib_x.npy')
f=x_data[0]
x_grid = np.linspace(0, 1, 601)
# f=np.square(x_grid)-x_grid
t_grid = np.linspace(0, 1, 201)
y_data=FD_CN_copy(f, x_grid, t_grid, 0.1)
X, T = np.meshgrid(x_grid, t_grid)
fig = plt.figure(figsize=(9,3),dpi=150)
ax1=fig.add_subplot(131,projection='3d')
ax1.plot_surface(T,X,y_data,cmap='rainbow')
ax1.set_xlabel('$t$')
ax1.set_ylabel('$x$')
ax1.set_title('$\epsilon=0.1$')
y_data=FD_CN_copy(f, x_grid, t_grid, 0.05)
ax2=fig.add_subplot(132,projection='3d')
ax2.plot_surface(T,X,y_data,cmap='rainbow')
ax2.set_xlabel('$t$')
ax2.set_ylabel('$x$')
ax2.set_title('$\epsilon=0.05$')
y_data=FD_CN_copy(f, x_grid, t_grid, 0.001)
ax3=fig.add_subplot(133,projection='3d')
ax3.plot_surface(T,X,y_data,cmap='rainbow')
ax3.set_xlabel('$t$')
ax3.set_ylabel('$x$')
ax3.set_zlabel('$u(x,t)$')
ax3.set_title('$\epsilon=0.001$')



plt.show(block=True)
