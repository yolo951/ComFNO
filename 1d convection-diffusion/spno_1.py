import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam
from scipy import integrate
import importlib
import SPNO_1d
# from SPNO_1d import *
importlib.reload(SPNO_1d)
from scipy.stats import multivariate_normal


x_data = np.load('spde_1_f.npy')
y_data = np.load('spde_1_u.npy')
x_grid = np.linspace(0, 1, x_data.shape[-1])

eps=0.001
ntrain = 900
ntest = 100

batch_size = 30
learning_rate = 0.001

epochs = 1000
step_size = 50
gamma = 0.5

modes1 = 20
modes2 = 20
interpolation_dim=201
width = 64

x_train = x_data[:ntrain, ::2]
y_train = y_data[:ntrain, ::2]
x_test = x_data[-ntest:, ::2]
y_test = y_data[-ntest:, ::2]
x_grid = x_grid[::2]

x_train = torch.Tensor(x_train)
y_train = torch.Tensor(y_train)
x_test = torch.Tensor(x_test)
y_test = torch.Tensor(y_test)

x_train = torch.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = torch.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size,
                                           shuffle=True)

model = SPNO_1d.SPNO1d(modes1, modes2, interpolation_dim,width, eps,x_grid.shape[-1]).cuda()
print('Total parameters:', count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

start = default_timer()

MSE = torch.zeros(epochs)
L2 = torch.zeros(epochs)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x)
        mse = F.mse_loss(out.view(batch_size, -1), y.view(batch_size, -1), reduction='mean')
        l2 = myloss(out.view(batch_size, -1), y.view(batch_size, -1))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()

    train_mse /= len(train_loader)
    train_l2 /= ntrain
    t2 = default_timer()

    MSE[ep] = train_mse
    L2[ep] = train_l2
    print('\repoch {:d}/{:d} L2 = {:.6f}, MSE = {:.6f}, using {:.6f}s'.format(ep + 1, epochs, train_l2, train_mse,
                                                                              t2 - t1), end='\n', flush=True)

print('Total training time:', default_timer() - start, 's')
pred = torch.zeros(y_test.shape)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=1, shuffle=False)
index = 0
test_l2 = 0
test_mse = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        out = model(x).view(-1)
        pred[index] = out
        mse = F.mse_loss(out.view(1, -1), y.view(1, -1), reduction='mean')
        test_l2 += myloss(out.view(1, -1), y.view(1, -1)).item()
        test_mse += mse.item()
        index += 1

    test_mse /= len(test_loader)
    test_l2 /= ntest
    print('test error: L2 =', test_l2, ', MSE =', test_mse)
residual = pred - y_test
fig = plt.figure(figsize=(8, 6), dpi=150)

plt.subplot(2, 2, 1)
plt.title("ground truth")
for i in range(ntest):
    plt.plot(x_grid, y_test[i])
plt.xlabel("x")
plt.ylabel("$u_g$")
plt.grid()

plt.subplot(2, 2, 2)
plt.title("prediction")
for i in range(ntest):
    plt.plot(x_grid, pred[i])
plt.xlabel("x")
plt.ylabel("$u_p$")
plt.grid()

plt.subplot(2, 2, 3)
plt.title("residuals")
plt.ylim([-0.01, 0.01])
for i in range(ntest):
    plt.plot(x_grid, residual[i])
plt.xlabel("x")
plt.ylabel("$u_g$-$u_p$")
plt.grid()

plt.subplot(2, 2, 4)
plt.title("training loss")
plt.plot(MSE, c='r', label='MSE')
plt.plot(L2, c='b', label='Relative')
plt.legend()
plt.yscale('log')
plt.xlabel("epoch")
plt.ylabel("error")
plt.grid()

plt.tight_layout()
plt.show(block=True)
v = torch.zeros(ntest)
m = torch.zeros(ntest)
for i in range(ntest):
    v[i] = torch.var(residual[i],unbiased=False)
    m[i] = torch.mean(residual[i])
print(torch.mean(m),torch.mean(v))
mse = MSE.numpy().reshape(1,MSE.shape[0])
l2 = L2.numpy().reshape(1,L2.shape[0])
loss = np.concatenate((mse,l2))
# np.save("spno_loss_1.npy",loss)
# torch.save(model,"spno_1")