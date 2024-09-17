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


# Gaussian Progress Regression
class GP_regression:
    def __init__(self, num_x_samples):
        self.observations = {"x": list(), "y": list()}
        self.num_x_samples = num_x_samples
        self.x_samples = np.linspace(0, 1.0, self.num_x_samples).reshape(-1, 1)

        self.mu = np.zeros_like(self.x_samples)
        self.cov = self.kernel(self.x_samples, self.x_samples)

    def update(self, observations):
        self.update_observation(observations)

        x = np.array(self.observations["x"]).reshape(-1, 1)
        y = np.array(self.observations["y"]).reshape(-1, 1)

        K11 = self.cov
        K22 = self.kernel(x, x)
        K12 = self.kernel(self.x_samples, x)
        K21 = self.kernel(x, self.x_samples)
        K22_inv = np.linalg.inv(K22 + 1e-8 * np.eye(len(x)))

        self.mu = K12.dot(K22_inv).dot(y)
        self.cov = self.kernel(self.x_samples, self.x_samples) - K12.dot(K22_inv).dot(K21)

    def visualize(self, num_gp_samples=3):
        gp_samples = np.random.multivariate_normal(
            mean=self.mu.ravel(),
            cov=self.cov,
            size=num_gp_samples)
        x_sample = self.x_samples.ravel()
        mu = self.mu.ravel()
        # uncertainty = 1.96 * np.sqrt(np.diag(self.cov))

        plt.figure()
        # plt.fill_between(x_sample, mu + uncertainty, mu - uncertainty, alpha=0.1)
        plt.plot(x_sample, mu, label='Mean')
        for i, gp_sample in enumerate(gp_samples):
            plt.plot(x_sample, gp_sample, lw=1, ls='-', label=f'Sample {i + 1}')

        plt.plot(self.observations["x"], self.observations["y"], 'rx')
        # plt.legend()
        plt.grid()
        return gp_samples

    def update_observation(self, observations):
        for x, y in zip(observations["x"], observations["y"]):
            if x not in self.observations["x"]:
                self.observations["x"].append(x)
                self.observations["y"].append(y)

    @staticmethod
    def kernel(x1, x2, l=0.5, sigma_f=0.2):
        dist_matrix = np.sum(x1 ** 2, 1).reshape(-1, 1) + np.sum(x2 ** 2, 1) - 2 * np.dot(x1, x2.T)
        return sigma_f ** 2 * np.exp(-0.5 / l ** 2 * dist_matrix)
# Finite difference method for singularly perturbed differential equations with initial-boundary problems
def FD_ib(f, grid_1, grid_2, eps=1):
    b_1 = f[0]
    b_2 = f[-1]
    dim_1 = grid_1.shape[0]
    dim_2 = grid_2.shape[0] + 1
    h_1 = 1 / (dim_1 - 1)
    h_2 = 1 / (dim_2 - 2)

    p = -eps / (h_1 ** 2)
    r = 2 * eps / (h_1 ** 2) - 1 / h_1 + 1 / h_2
    q = -eps / (h_1 ** 2) + 1 / h_1
    o = -1 / h_2
    s = 0

    res = np.zeros((dim_2 - 1, dim_1))
    res[0, :] = f[:]
    for i in range(dim_2 - 1):
        res[i, 0] = f[0]
        res[i, -1] = f[-1]

    U = np.zeros((dim_1 - 2, dim_1 - 2))
    U[0, :2] = np.array([r, q])
    U[-1, -2:] = np.array([p, r])

    j = 0
    for i in range(1, dim_1 - 3):
        U[i, j:j + 3] = np.array([p, r, q])
        j += 1

    T = np.eye(dim_1 - 2) * o
    V = np.eye(dim_1 - 2) * s
    Z = np.zeros((dim_1 - 2, dim_1 - 2))

    A_blc = np.empty((dim_2 - 2, dim_2 - 2), dtype=object)
    for i in range(dim_2 - 2):
        for j in range(dim_2 - 2):
            if i == j:
                A_blc[i, j] = U
            elif i + 1 == j:
                A_blc[i, j] = V
            elif i - 1 == j:
                A_blc[i, j] = T
            else:
                A_blc[i, j] = Z
    A = np.vstack([np.hstack(A_i) for A_i in A_blc])

    B = np.zeros((dim_2 - 2, dim_1 - 2))
    B[0, :] = f[1:-1]
    B = np.reshape(B, ((dim_2 - 2) * (dim_1 - 2), 1)) * (-o)

    C = np.zeros((dim_2 - 2, dim_1 - 2))
    for i in range(dim_2 - 2):
        C[i, 0] = f[0]
    C = np.reshape(C, ((dim_2 - 2) * (dim_1 - 2), 1)) * (-p)

    D = np.zeros((dim_2 - 2, dim_1 - 2))
    for i in range(dim_2 - 2):
        D[i, -1] = f[-1]
    D = np.reshape(D, ((dim_2 - 2) * (dim_1 - 2), 1)) * (-q)

    sol = np.linalg.solve(A, B + C + D)
    sol = np.reshape(sol, (dim_2 - 2, dim_1 - 2))
    res[1:, 1:-1] = sol

    return res
# approximate exact solution of 1d singularly perturbed equ
# u_t-eps*u''+b(x)u'+c(x)u=f(x,t), u(0,t)=u(1,t)=0, u(x,0)=g(x)
# by finite difference method on Shishkin mesh.
def FD_CN_ib(g, gridx ,gridt, epsilon):
    b=lambda x: np.ones_like(x)
    c=lambda x: x
    f=lambda x,t: np.zeros_like(x)


    alpha = 1
    Nx = gridx.shape[0]
    Nt = gridt.shape[0]
    Nd = g.shape[0]
    sigma = min(1 / 2, epsilon * np.log(Nx-1) / alpha)
    gridS = np.hstack(
        (np.linspace(0, 1-sigma, int((Nx - 1) / 2) + 1), np.linspace(1-sigma, 1, int((Nx - 1) / 2) + 1)[1:]))
    gS = interpolate.interp1d(gridx, g)(gridS)
    h1= (1 - sigma) / ((Nx - 1) / 2)
    h2 = sigma / ((Nx - 1) / 2)
    dt=1/(Nt-1)
    yS = np.zeros((Nd, Nx))
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
    for k in range(Nd):
        yS[k]=gS[k]
        for j in range(Nt-1):
            B1=0.5*dt*f(gridS,(j+1)*dt)+0.5*dt*f(gridS,j*dt)
            B=np.dot(np.eye(Nx-2)-0.5*dt*U,yS[k,1:-1])+B1[1:-1]

            yS[k, 0] = 0
            yS[k, 1:-1] = np.linalg.solve(A, B).flatten()
            yS[k, -1] = 0
    y = interpolate.interp1d(gridS, yS)(gridx)
    return y


# generate random functions(1d) default dim=1001
def generate(samples=1000, begin=0, end=1, random_dim=101, out_dim=1001, length_scale=1, interp="cubic"):
    space = GaussianRandomField(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples)
    x = np.linspace(begin, end, out_dim)
    f = space.eval_u(features, x[:, None])
    return f


# spde_initial-boundary problem
gp1 = GP_regression(num_x_samples=601)
gp1.update({"x": [0, 1], "y": [1, 0]})
x_data = gp1.visualize(100)
# x_data=np.load('spde_ib_x.npy')

x_grid = np.linspace(0, 1, 601)
N = x_data.shape[0]
t_grid = np.linspace(0, 1, 201)
# y_data=FD_CN_ib(x_data, x_grid, t_grid, 0.001)
# np.save('spde_ib_x.npy', x_data)
# np.save('spde_ib_y.npy', y_data)
fig=plt.figure()
for i in range(100):
    plt.plot(x_grid, x_data[i].flatten())
plt.grid()
# plt.legend()
plt.show(block=True)
# X, T = np.meshgrid(x_grid, t_grid)
# fig = plt.figure(figsize=(8, 3), dpi=150)
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.set_title("$u_t-\epsilon u_{xx}+u_x=0$")
# im1 = ax1.pcolormesh(T, X, y_data[0], shading='auto')
# ax1.set_xlabel("t")
# ax1.set_ylabel("x")
# fig.colorbar(im1)
#
# ax2 = fig.add_subplot(1, 2, 2, projection='3d')
# ax2.set_title("$u_t-\epsilon u_{xx}+u_x=0$ (3d)")
# surf = ax2.plot_surface(T, X, y_data[0], rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'))
# ax2.set_xlabel("t")
# ax2.set_ylabel("x")
# ax2.set_zlabel("u")
# fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.2)
# plt.tight_layout()
# plt.show(block=True)
# fig.savefig("ib_example.png")

