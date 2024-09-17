import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import interpolate
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import operator
from functools import reduce
from functools import partial
from timeit import default_timer
from utilities3 import *
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SPNO2d(nn.Module):
    def __init__(self, modes1, modes2, width1, width2, interpolation_dim, eps, x_grid_dim, y_grid_dim, x_begin=0, x_end=1, y_begin=0,
                 y_end=1):
        super(SPNO2d, self).__init__()

        self.x_begin = x_begin
        self.x_end = x_end
        self.y_begin = y_begin
        self.y_end = y_end
        self.modes1 = modes1
        self.modes2 = modes2
        self.width1 = width1
        self.width2 = width2
        self.eps = eps
        self.x_grid_dim = x_grid_dim
        self.y_grid_dim = y_grid_dim
        self.interpolation_dim=interpolation_dim

        self.padding = 9  # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width1)  # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.width1, self.width1, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width1, self.width1, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width1, self.width1, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width1, self.width1, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d(self.width1, self.width1, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width1, self.width1, 1)
        self.w1 = nn.Conv2d(self.width1, self.width1, 1)
        self.w2 = nn.Conv2d(self.width1, self.width1, 1)
        self.w3 = nn.Conv2d(self.width1, self.width1, 1)
        self.w4 = nn.Conv2d(self.width1, self.width1, 1)

        self.fc1 = nn.Linear(self.width1, 64)
        self.fc2 = nn.Linear(64, 1)

        self.f1 = nn.Linear(self.interpolation_dim, 128)
        self.f2 = nn.Linear(128, 1)
        self.f3 = nn.Linear(self.interpolation_dim, 128)
        self.f4 = nn.Linear(128, 2)

        self.fi0 = nn.Linear(3, self.width2)
        self.fi1 = nn.Linear(3, self.width2)
        self.fi00 = nn.Linear(1, 1)
        self.fi01 = nn.Linear(2, 2)
        self.fi10 = nn.Linear(1, 1)
        self.fi11 = nn.Linear(2, 2)

        self.coni0 = SpectralConv2d(self.width2, self.width2, self.modes1, self.modes2)
        self.coni1 = SpectralConv2d(self.width2, self.width2, self.modes1, self.modes2)
        self.coni2 = SpectralConv2d(self.width2, self.width2, self.modes1, self.modes2)
        self.coni3 = SpectralConv2d(self.width2, self.width2, self.modes1, self.modes2)

        self.wi0 = nn.Conv2d(self.width2, self.width2, 1)
        self.wi1 = nn.Conv2d(self.width2, self.width2, 1)
        self.wi2 = nn.Conv2d(self.width2, self.width2, 1)
        self.wi3 = nn.Conv2d(self.width2, self.width2, 1)

        self.fi3 = nn.Linear(self.width2, 64)
        self.fi4 = nn.Linear(64, 1)
        self.fi5 = nn.Linear(self.width2, 64)
        self.fi6 = nn.Linear(64, 1)

    def forward(self, x):

        x_n = torch.reshape(x, (x.shape[0], x.shape[1],x.shape[2]))
        gridx = torch.tensor(np.linspace(self.x_begin, self.x_end, x.shape[1]), dtype=torch.float)
        gridy = torch.tensor(np.linspace(self.y_begin, self.y_end, x.shape[2]), dtype=torch.float)
        grid = torch.tensor(np.linspace(self.x_begin, self.x_end, self.interpolation_dim), dtype=torch.float)
        x_n = x_n.cpu()
        gridx = gridx.cpu()
        gridy = gridy.cpu()
        grid=grid.cpu()
        x_n=torch.stack([torch.tensor(interpolate.RectBivariateSpline(gridx, gridy, x_n[i], kx=3, ky=3)(grid, grid),dtype=torch.float) for i in range(x.shape[0])])
        x_n = x_n.to(x.device)

        grid = self.get_grid(x.shape, x.device)

        # x_n = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2]))
        c = self.f1(x_n)
        c = F.gelu(c)
        c = self.f2(c)
        c = torch.reshape(c, (c.shape[0], c.shape[1]))
        c = self.f3(c)
        c = F.gelu(c)
        c = self.f4(c)

        c1 = c[:, 0][:, None]
        c2 = c[:, 1][:, None]


        x_1 = torch.cat((self.fi00(x), self.fi01(-(grid - 0) / self.eps)), dim=-1)
        x_1 = self.fi0(x_1)
        x_1 = x_1.permute(0, 3, 1, 2)
        x_1 = self.coni0(x_1) + self.wi0(x_1)
        x_1 = F.gelu(x_1)
        x_1 = self.coni1(x_1) + self.wi1(x_1)
        x_1 = x_1.permute(0, 2, 3, 1)
        x_1 = self.fi3(x_1)
        x_1 = F.gelu(x_1)
        x_1 = self.fi4(x_1)
        x_1 = self.mul(torch.exp(x_1), c1)

        x_2 = torch.cat((self.fi10(x), self.fi11((grid - 1) / self.eps)), dim=-1)
        x_2 = self.fi1(x_2)
        x_2 = x_2.permute(0, 3, 1, 2)
        x_2 = self.coni2(x_2) + self.wi2(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.coni3(x_2) + self.wi3(x_2)
        x_2 = x_2.permute(0, 2, 3, 1)
        x_2 = self.fi5(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.fi6(x_2)
        x_2 = self.mul(torch.exp(x_2), c2)

        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv4(x)
        x2 = self.w4(x)
        x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x + x_1 + x_2

    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(self.x_begin, self.x_end, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(self.y_begin, self.y_end, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

    def get_1d_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.begin, self.end, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def mul(self, a, b):
        return torch.einsum("bmni,bi->bmni", a, b)