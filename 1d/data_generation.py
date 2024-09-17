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


# approximate exact solution of 1d singularly perturbed advection diffusion
# -eps*u''+pu'+qu=f, u(0)=u(1)=0
# by finite difference method on Shishkin mesh.
def FD_multi_eps(f, grid):
    def p(x):
        return x + 1  # p(x)>=alpha=1

    def q(x):
        return 0

    alpha = 1
    Nd = f.shape[0]
    N = grid.shape[-1]
    y = np.zeros((Nd, N))
    for k in range(Nd):
        epsilon = f[k, -1]
        sigma = min(1 / 2, 2 * epsilon * np.log(N - 1) / alpha)
        gridS = np.hstack(
            (np.linspace(0, 1 - sigma, int((N - 1) / 2) + 1), np.linspace(1 - sigma, 1, int((N - 1) / 2) + 1)[1:]))
        fS = interpolate.interp1d(grid, f[k,:-1])(gridS)
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
        B = np.zeros(N - 2)
        B[:] = fS[1:-1]
        B = B.T

        yS[k, 0] = 0
        yS[k, 1:-1] = np.linalg.solve(U, B).flatten()
        yS[k, -1] = 0
        y[k] = interpolate.interp1d(gridS, yS[k])(grid)
    return y

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


############## 1d time-dependent problem #####################
# gp1 = GP_regression(num_x_samples=601)
# gp1.update({"x": [0, 1], "y": [0, 0]})
# x_data = gp1.visualize(1000)
#
#
# x_grid = np.linspace(0, 1, 601)
# N = x_data.shape[0]
# t_grid = np.linspace(0, 1, 201)
# y_data=FD_CN_ib(x_data, x_grid, t_grid, 0.001)
# np.save('spde_ib_x.npy', x_data)
# np.save('spde_ib_y.npy', y_data)
# fig=plt.figure()
# for i in range(1000):
#     plt.plot(x_grid, y_data[i].flatten())
# plt.grid()
# plt.show(block=True)


################## multiple epsilon case #############
# x_data = generate(samples=10100)
# x_grid = np.linspace(0,1,x_data.shape[-1])
# eps = np.zeros((x_data.shape[0],1))
# for i in range(x_data.shape[0]):
#     eps[i,0] = (i%100+1)*0.001
# x_data = np.concatenate((x_data,eps),-1)
# y_data = FD_multi_eps(x_data,x_grid)
# np.save('spde_mul_x.npy',x_data)
# np.save('spde_mul_y.npy',y_data)
# for i in range(1000):
#     plt.title("$-\epsilon u''+(x+1)u'=f$")
#     plt.plot(x_grid,y_data[i])
#     plt.xlabel("x")
#     plt.ylabel("u")
# plt.grid()
# plt.show()


################# 1d turning points problem #############
# fig = plt.figure(dpi=150)
# f = generate(begin=-1, end=1)
#
# x,f,y_data = FD_tp(f,0.001, meshtype='Equal')
# np.save('spde_tp_f.npy',f)
# np.save('spde_tp_u.npy',y_data)
# plt.plot()
# for i in range(1000):
#     plt.title("$-\epsilon u''+x(x+1)u'+u=f$")
#     plt.plot(x,y_data[i])
#     plt.xlabel("x")
#     plt.ylabel("u")
# plt.grid()
# plt.show(block=True)