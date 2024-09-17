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


# Finite difference method for ordinary differential equations
def FD_AD_2d(f, epsilon, meshtype='Shishkin'):
    def p1(x,y):
        return 1  # p1(x), p2(x)>=alpha=1
    def p2(x,y):
        return 1
    def q(x,y):
        return 1
    alpha = 1


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
        fS[k] = interpolate.RectBivariateSpline(grid, grid, f[k],kx=1,ky=1)(gridS, gridS)
        B[:] = grid_to_vec(fS[k],N-2)
        X=np.linalg.solve(U, B).flatten()
        yS[k]=vec_to_grid(X,N)

        y[k] = interpolate.RectBivariateSpline(gridS, gridS,yS[k],kx=1,ky=1)(grid,grid)
    if meshtype == 'Shishkin':
        return yS
    else:
        return y
def vec_to_grid(x,N):
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
def grid_to_vec(X,N):
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


# generate random functions(1d) default dim=1001
def generate(samples=10, begin=0, end=1, random_dim=101, out_dim=101, length_scale=1, interp="cubic", A=0):
    space = GRF(begin, end, length_scale=length_scale, N=random_dim, interp=interp)
    features = space.random(samples, A)
    features=np.array([vec_to_grid(y,N=random_dim) for y in features])
    x_grid = np.linspace(begin, end, out_dim)
    x_data = space.eval_u(features, x_grid,x_grid)
    return x_data  # X_data.shape=(samples,out_dim,out_dim)


class GRF(object):
    def __init__(self, begin=0, end=1, length_scale=1, N=1000, interp="cubic"):
        self.N = N
        self.interp = interp
        self.x = np.linspace(begin, end, num=N)
        self.z=np.zeros((self.N**2,2))
        for j in range(self.N):
            for i in range(self.N):
                self.z[j*self.N+i]=[self.x[i],self.x[j]]
        self.K=np.exp(-0.5*self.distance_matrix(self.z,length_scale))
        self.L = np.linalg.cholesky(self.K + 1e-12 * np.eye(self.N**2))
    def distance_matrix(self,x,length_scale):
        n=x.shape[0]
        grid=np.zeros((n,n))
        for i in range(n):
            for j in range(i):
                grid[i][j]=((x[i][0]-x[j][0])**2+(x[i][1]-x[j][1])**2)/length_scale**2
                grid[j][i]=grid[i][j]
        return grid


    def random(self, n, A):
        """Generate `n` random feature vectors.
        """
        u = np.random.randn(self.N**2, n)
        return np.dot(self.L, u).T + A


    def eval_u(self, ys, x,y):
        """For a list of functions represented by `ys`,
        compute a list of a list of function values at a list `sensors`.
        """

        res = np.zeros((ys.shape[0], x.shape[0],x.shape[0]))
        for i in range(ys.shape[0]):
            res[i] = interpolate.RectBivariateSpline(self.x,self.x, ys[i], kx=3,ky=3)(
                x,y)
        return res



x_grid=np.linspace(0,1,101)
f = generate(samples=1000,out_dim=101)
y_data= FD_AD_2d(f,0.001,meshtype='Equal')


np.save('spde_2d_x.npy',f)
np.save('spde_2d_y.npy',y_data)

X, Y = np.meshgrid(x_grid, x_grid)
fig = plt.figure(figsize=(8,3),dpi=150)
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_title("$-\epsilon \Delta u+u_x+u_y=f$")
im1 = ax1.pcolormesh(X, Y, y_data[0], shading='auto')
ax1.set_xlabel("y")
ax1.set_ylabel("x")
fig.colorbar(im1)

ax2 = fig.add_subplot(1, 2, 2, projection='3d')
ax2.set_title("$-\epsilon \Delta u+u_x+u_y=f$ (3d)")
surf = ax2.plot_surface(X,Y,y_data[0],rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))
ax2.set_xlabel("y")
ax2.set_ylabel("x")
ax2.set_zlabel("u")
fig.colorbar(surf, shrink=0.5, aspect=10,pad=0.2)
plt.tight_layout()
plt.show(block=True)
# fig.savefig("2d_example.png")
