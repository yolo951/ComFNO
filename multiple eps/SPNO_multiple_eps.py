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
from scipy import interpolate
from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)


################################################################
#  1d fourier layer
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


class SPNO1d(nn.Module):
    def __init__(self, modes, width, grid_dim, begin=0, end=1):
        super(SPNO1d, self).__init__()
        self.begin = begin
        self.end = end
        self.modes1 = modes
        self.width = width
        self.grid_dim = grid_dim
        self.padding = 2  # pad the domain if input is non-periodic
        # input channel is 2: (a(x), x)
        self.fc0 = nn.Linear(2, self.width)
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)

        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)
        self.f3 = nn.Linear(self.grid_dim, 64)
        self.f4 = nn.Linear(64, 1)


        self.fi1 = nn.Linear(2, self.width)
        self.fi2 = nn.Linear(self.width, self.width)


        self.d11 = nn.Linear(self.grid_dim, 64)
        self.d12 = nn.Linear(64, self.grid_dim)
        self.d21 = nn.Linear(1, 64)
        self.d22 = nn.Linear(64, 1)


        self.coni2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.coni3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.coni4 = SpectralConv1d(self.width, self.width, self.modes1)

        self.wi2 = nn.Conv1d(self.width, self.width, 1)
        self.wi3 = nn.Conv1d(self.width, self.width, 1)
        self.wi4 = nn.Conv1d(self.width, self.width, 1)
        self.fi5 = nn.Linear(self.width, 128)
        self.fi6 = nn.Linear(128, 1)

    def forward(self, x):
        eps = x[:, -1, :]
        x = x[:, :-1, :]
        x_n = torch.reshape(x, (x.shape[0], x.shape[1]))
        grid = self.get_grid(x.shape, x.device)
        c2 = self.f3(x_n)
        c2 = F.gelu(c2)
        c2 = self.f4(c2)



        x_21=self.d11(x_n)
        x_21=F.gelu(x_21)
        x_21=self.d12(x_21)
        x_22=self.mul((grid - 1), 1 / eps)
        x_22=self.d21(x_22)
        x_22=F.gelu((x_22))
        x_22=self.d22(x_22)
        x_2=torch.cat((x_21[:,:,None],x_22),dim=-1)


        x_2 = self.fi1(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.fi2(x_2)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = self.coni2(x_2) + self.wi2(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.coni3(x_2) + self.wi3(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.coni4(x_2) + self.wi4(x_2)
        x_2 = x_2.permute(0, 2, 1)
        x_2 = self.fi5(x_2)
        x_2 = F.gelu(x_2)
        x_2 = self.fi6(x_2)


        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        # x = F.pad(x, [0,self.padding]) # pad the domain if input is non-periodic

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


        # x = x[..., :-self.padding] # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

        return x + self.mul(torch.exp(x_2), c2)#+ self.mul(torch.exp(x_1), c1)

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(self.begin, self.end, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)

    def mul(self, a, b):
        return torch.einsum("bmi,bi->bmi", a, b)