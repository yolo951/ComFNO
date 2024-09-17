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


# approximate exact solution of 1d singularly perturbed advection diffusion
# -eps*u''+pu'+qu=f, u(0)=u(1)=0
# by finite difference method on Shishkin mesh.
def FD_AD_1d(f, epsilon, meshtype='Shishkin'):
    def p(x):
        return x+1 # p(x)>=alpha=1

    def q(x):
        return 0

    alpha = 1
    N = f.shape[-1]
    Nd = f.shape[0]
    sigma = min(1 / 2, 2*epsilon * np.log(N-1) / alpha)
    grid = np.linspace(0, 1, N)
    gridS = np.hstack(
        (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
    fS = interpolate.interp1d(np.linspace(0, 1, N), f)(gridS)
    h1 = (1 - sigma) / ((N - 1) / 2)
    h2 = sigma / ((N - 1) / 2)
    yS = np.zeros((Nd, N))
    U = np.zeros((N - 2, N - 2))
    U[0, :2] = np.array([2 * epsilon / (h1 ** 2) + p(h1) / h1 + q(h1), -epsilon / (h1 ** 2)])
    U[-1, -2:] = np.array(
        [-epsilon / (h2 ** 2) - p(1 - h2) / h2, 2 * epsilon / (h2 ** 2) + p(1 - h2) / h2 + q(1 - h2)])

    Nm = int((N - 3) / 2)  # here N must be odd
    for i in range(1, Nm):
        x_i = (i + 1) * h1
        p1 = -epsilon / (h1 ** 2) - p(x_i) / h1
        r1 = 2 * epsilon / (h1 ** 2) + p(x_i) / h1 + q(x_i)
        q1 = -epsilon / (h1 ** 2)
        U[i, i - 1:i + 2] = np.array([p1, r1, q1])
    x_i = (Nm + 1) * h1
    U[Nm, Nm - 1:Nm + 2] = np.array(
        [-2 * epsilon / (h1 * (h1 + h2)) - p(x_i) / h1, 2 * epsilon / (h1 * h2) + p(x_i) / h1 + q(x_i),
         -2 * epsilon / (h2 * (h1 + h2))])
    for i in range(Nm + 1, N - 3):
        x_i = (Nm + 1) * h1 + (i - Nm) * h2
        p2 = -epsilon / (h2 ** 2) - p(x_i) / h2
        r2 = 2 * epsilon / (h2 ** 2) + p(x_i) / h2 + q(x_i)
        q2 = -epsilon / (h2 ** 2)
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
f = generate()

x,f,y_data = FD_AD_1d(f,0.001, meshtype='Equal')
np.save('spde_1_f.npy',f)
np.save('spde_1_u.npy',y_data)
plt.plot()
for i in range(1000):
    plt.title("$-\epsilon u''+(x+1)u'+u=f$")
    plt.plot(x,y_data[i])
    plt.xlabel("x")
    plt.ylabel("u")
plt.grid()
plt.show(block=True)
