import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy import integrate
from scipy import linalg
from scipy import interpolate
from sklearn import gaussian_process as gp
from mpl_toolkits.mplot3d import Axes3D


# Gaussian Random Field
class GaussianRandomField(object):
    def __init__(self, begin=0, end=1, kernel="RBF", length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)[:, None]
        if kernel == "RBF":
            K = gp.kernels.RBF(length_scale=length_scale)
        elif kernel == "AE":
            K = gp.kernels.Matern(length_scale=length_scale, nu=0.5)
        self.K = K(self.x)
        self.L = np.linalg.cholesky(self.K + 1e-13 * np.eye(self.N))

    def random(self, n):
        u = np.random.randn(self.N, n)
        return np.dot(self.L, u).T

    def eval_u_one(self, y, x):
        if self.interp == "linear":
            return np.interp(x, np.ravel(self.x), y)
        f = interpolate.interp1d(
            np.ravel(self.x), y, kind=self.interp, copy=False, assume_sorted=True
        )
        return f(x)

    def eval_u(self, ys, sensors):
        if self.interp == "linear":
            return np.vstack([np.interp(sensors, np.ravel(self.x), y).T for y in ys])

        res = np.zeros((ys.shape[0], sensors.shape[0]))
        for i in range(ys.shape[0]):
            res[i, :] = interpolate.interp1d(np.ravel(self.x), ys[i], kind=self.interp, copy=False, assume_sorted=True)(
                sensors).T
        return res


# approximate exact solution of 1d singularly perturbed turning point problem
# -eps*u''+bu'+cu=f, u(-1)=u(1)=0
# by finite difference method on Shishkin mesh.
def FD_tp(f, epsilon, meshtype='Shishkin'):
    def b(x):
        return x*(x+2) # p(x)>=alpha=1

    def c(x):
        return 1
    #(N-1)%4==0
    alpha = 1
    N = f.shape[-1]
    Nd = f.shape[0]
    sigma = min(1 / 2, epsilon * np.log(N-1) / alpha)
    grid = np.linspace(-1, 1, N)
    gridS = np.hstack(
        (np.linspace(-1, -1 + sigma, int((N - 1) / 4) + 1),np.linspace(-1 + sigma, 1 - sigma, int((N - 1) / 2) + 1)[1:], np.linspace(1 - sigma, 1, int((N - 1) / 4) + 1)[1:]))
    fS = interpolate.interp1d(grid, f)(gridS)
    h1 = 4*sigma / (N - 1)
    h2 = (4-4*sigma) / (N - 1)
    yS = np.zeros((Nd, N))
    U = np.zeros((N - 2, N - 2))
    U[0, :2] = np.array([2 * epsilon / (h1 ** 2) - b(-1+h1) / h1 + c(-1+h1), -epsilon / (h1 ** 2)+b(-1+h1)/h1])
    U[-1, -2:] = np.array(
        [-epsilon / (h1 ** 2) - b(1 - h1) / h1, 2 * epsilon / (h1 ** 2) + b(1 - h1) / h1 + c(1 - h1)])

    N1=int((N-1)/4-1)
    for i in range(1,N1):
        x_i = -1+(i + 1) * h1
        p1 = -epsilon / (h1 ** 2)
        r1 = 2 * epsilon / (h1 ** 2) - b(x_i) / h1 + c(x_i)
        q1 = -epsilon / (h1 ** 2) + b(x_i) / h1
        U[i, i - 1:i + 2] = np.array([p1, r1, q1])
    x_i = -1+sigma
    U[N1, N1 - 1:N1 + 2] = np.array(
        [-2 * epsilon / (h1 * (h1 + h2)) , 2 * epsilon / (h1 * h2) - b(x_i) / h2 + c(x_i),
         -2 * epsilon / (h2 * (h1 + h2))+ b(x_i) / h2])
    N2=int((N - 3) / 2)
    for i in range(N1+1, N2+1):
        x_i = -1+sigma + (i - N1) * h2
        p2 = -epsilon / (h2 ** 2)
        r2 = 2 * epsilon / (h2 ** 2) - b(x_i) / h2 + c(x_i)
        q2 = -epsilon / (h2 ** 2) + b(x_i) / h2
        U[i, i - 1:i + 2] = np.array([p2, r2, q2])
    N3 = int(3*(N-1)/4-1)
    for i in range(N2+1, N3):
        x_i = (i - N2) * h2
        p1 = -epsilon / (h2 ** 2) - b(x_i) / h2
        r1 = 2 * epsilon / (h2 ** 2) + b(x_i) / h2 + c(x_i)
        q1 = -epsilon / (h2 ** 2)
        U[i, i - 1:i + 2] = np.array([p1, r1, q1])
    x_i = 1-sigma
    U[N3, N3 - 1:N3 + 2] = np.array(
        [-2 * epsilon / (h2 * (h1 + h2)) - b(x_i) / h2, 2 * epsilon / (h1 * h2) + b(x_i) / h2 + c(x_i),
         -2 * epsilon / (h1 * (h1 + h2))])
    for i in range(N3 + 1, N - 3):
        x_i = 1-sigma + (i - N3) * h1
        p2 = -epsilon / (h1 ** 2) - b(x_i) / h1
        r2 = 2 * epsilon / (h1 ** 2) + b(x_i) / h1 + c(x_i)
        q2 = -epsilon / (h1 ** 2)
        U[i, i - 1:i + 2] = np.array([p2, r2, q2])
    for k in range(Nd):
        B = np.zeros(N - 2)
        B[:] = fS[k, 1:-1]
        B = B.T

        yS[k, 0] = 0
        yS[k, 1:-1] = np.linalg.solve(U, B).flatten()
        yS[k, -1] = 0
    y = interpolate.interp1d(gridS, yS)(grid)
    if meshtype == 'Shishkin':
        return gridS,fS,yS
    else:
        return grid,f,y



# generate random functions(1d) default dim=1001
def generate(samples=1000, begin=0, end=1, random_dim=101, out_dim=1001, length_scale=1, interp="cubic"):
    space = GaussianRandomField(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples)
    x = np.linspace(begin, end, out_dim)
    f = space.eval_u(features, x[:, None])
    return f

fig = plt.figure(dpi=150)
f = generate(begin=-1, end=1)

x,f,y_data = FD_tp(f,0.001, meshtype='Equal')
# np.save('spde_tp_f.npy',f)
# np.save('spde_tp_u.npy',y_data)
plt.plot()
for i in range(1000):
    plt.title("$-\epsilon u''+x(x+1)u'+u=f$")
    plt.plot(x,y_data[i])
    plt.xlabel("x")
    plt.ylabel("u")
plt.grid()
plt.show(block=True)
