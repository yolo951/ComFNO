import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D
from data_generation import generate

def FD_AD_2d(f, epsilon, meshtype='Shishkin'):
    def p1(x,y):
        return 1  # p(x)>=alpha=1
    def p2(x,y):
        return 1
    def q(x,y):
        return 1
    alpha = 1#p1和p2共同的下界


    N = f.shape[-1]
    Nd = f.shape[0]
    sigma = min(1 / 2, epsilon * np.log(N) / alpha)
    grid = np.linspace(0, 1, N)
    gridS = np.hstack(
        (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
    fS=np.zeros((Nd,N,N))
    h1 = (1 - sigma) / ((N - 1) / 2)
    h2 = sigma / ((N - 1) / 2)
    yS = np.zeros((Nd, N, N))
    y = np.zeros((Nd, N, N))

    for k in range(Nd):

        U = np.zeros(((N-2)**2,(N-2)**2))
        def each_row_of_U(i, hx_1, hx_2, hy_1, hy_2, x, y):
            U[i, i] =2*epsilon/hx_1/hx_2+2*epsilon/hy_1/hy_2+p1(x,y)/hx_1+p2(x,y)/hy_1+q(x,y)
            if i%(N-2)>0:
                U[i, i-1] =-2*epsilon/hx_1/(hx_1+hx_2)-p1(x,y)/hx_1
            if i-(N-2)>=0:
                U[i, i-(N-2)] =-2*epsilon/hy_1/(hy_1 + hy_2)-p2(x, y)/hy_1
            if i%(N-2)<N-3:
                U[i, i+1] =-2*epsilon/hx_2/(hx_1 + hx_2)
            if i+N-2<(N-2)**2:
                U[i, i+N-2] =-2*epsilon/hy_2/(hy_1+hy_2)
        Nm = int((N-3)/2)  # here N must be odd
        for j in range(Nm):
            y_j = (j + 1) * h1
            for i in range(Nm):
                n=j*(N-2)+i
                x_i =(i+1)*h1
                each_row_of_U(n,h1,h1,h1,h1,x_i,y_j)
            n=j*(N-2)+Nm
            x_i=(Nm+1)*h1
            each_row_of_U(n,h1,h2,h1,h1,x_i,y_j)
            for i in range(Nm+1,N-2):
                n = j*(N-2)+i
                x_i=(Nm+1)*h1+(i-Nm)*h2
                each_row_of_U(n,h2,h2,h1,h1,x_i,y_j)
        y_j = (Nm + 1) * h1
        for i in range(Nm):
            n = Nm*(N-2)+i
            x_i = (i+1)*h1
            each_row_of_U(n,h1,h1,h1,h2,x_i,y_j)
        n = Nm * (N - 2) + Nm
        x_i = (Nm + 1) * h1
        each_row_of_U(n,h1,h2,h1,h2,x_i,y_j)
        for i in range(Nm+1, N-2):
            n = Nm*(N-2)+i
            x_i =(Nm+1)*h1+(i-Nm)*h2
            each_row_of_U(n,h2,h2,h1,h2,x_i,y_j)
        for j in range(Nm+1,N-2):
            y_j =(Nm+1)*h1+(j-Nm)*h2
            for i in range(Nm):
                n=j*(N-2)+i
                x_i =(i+1)*h1
                each_row_of_U(n,h1,h1,h2,h2,x_i,y_j)
            n=j*(N-2)+Nm
            x_i=(Nm+1)*h1
            each_row_of_U(n,h1,h2,h2,h2,x_i,y_j)
            for i in range(Nm+1,N-2):
                n = j*(N-2)+i
                x_i=(Nm+1)*h1+(i-Nm)*h2
                each_row_of_U(n,h2,h2,h2,h2,x_i,y_j)
        B = np.zeros(((N - 2)**2,1))
        fS[k] = interpolate.RectBivariateSpline(grid, grid, f[k],kx=1,ky=1)(gridS, gridS)  # 默认为线性插值
        B[:] = grid_to_vec(fS[k],N-2)
        X=np.linalg.solve(U, B).flatten()
        yS[k]=vec_to_grid(X,N)

        y[k] = interpolate.RectBivariateSpline(gridS, gridS,yS[k],kx=1,ky=1)(grid,grid)
    if meshtype == 'Shishkin':
        return yS
    else:
        return y
def vec_to_grid(x,N):#返回size为(N,N)的数组
    res = np.zeros((N, N))
    if N**2==x.shape[0]:
        for i in range(N):
            res[i]=x[i*N:(i+1)*N].T
    elif N**2>x.shape[0]:
        for i in range(1,N-1):
            res[i,1:-1]=x[(i-1)*(N-2):i*(N-2)].T
    else:
        for i in range(N):
            res[i]= x[(i+1)*(N+2)+1:(i+2)*(N+2)-1].T
    return res
def grid_to_vec(X,N):#返回size为(N**2,1)的数组
    n=X.shape[0]
    res = np.zeros((N ** 2, 1))
    if n==N:
        for i in range(N):
            res[i*N:(i+1)*N]=X[i][:,None]
    elif n==N+2:
        for i in range(N):
            res[i*N:(i+1)*N]=X[i+1,1:-1][:,None]
    elif N==n+2:
        for i in range(1,N-1):
            res[i*N+1:(i+1)*N-1]=X[i-1][:,None]
    return res


f=generate(samples=1,out_dim=201)
x_grid=np.linspace(0,1,201)
y_data=FD_AD_2d(f, 0.1,meshtype='Equal')
y_data=y_data.squeeze()
X, Y = np.meshgrid(x_grid, x_grid)
fig = plt.figure(figsize=(9,3),dpi=150)
ax1=fig.add_subplot(131,projection='3d')
ax1.plot_surface(Y,X,y_data,cmap='rainbow')
ax1.set_xlabel('$y$')
ax1.set_ylabel('$x$')
ax1.set_title('$\epsilon=0.1$')
y_data=FD_AD_2d(f, 0.05,meshtype='Equal')
y_data=y_data.squeeze()
ax2=fig.add_subplot(132,projection='3d')
ax2.plot_surface(Y,X,y_data,cmap='rainbow')
ax2.set_xlabel('$y$')
ax2.set_ylabel('$x$')
ax2.set_title('$\epsilon=0.05$')
y_data=FD_AD_2d(f, 0.001,meshtype='Equal')
y_data=y_data.squeeze()
ax3=fig.add_subplot(133,projection='3d')
ax3.plot_surface(Y,X,y_data,cmap='rainbow')
ax3.set_xlabel('$y$')
ax3.set_ylabel('$x$')
ax3.set_zlabel('$u(x,y)$')
ax3.set_title('$\epsilon=0.001$')



plt.show(block=True)
