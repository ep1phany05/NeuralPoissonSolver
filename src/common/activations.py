# https://github.com/kwea123/Coordinate-MLPs/blob/master/models.py

import torch
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=30.0):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        self.w0 = self.w0.to(input.device)
        return torch.sin(self.w0 * input)


class ExpSine(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.sin(self.a * x))


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input) - self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.sigmoid(input)


class Gaussian(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x ** 2 / (2 * self.a ** 2))


class Quadratic(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2)


class MultiQuadratic(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return 1 / (1 + (self.a * x) ** 2) ** 0.5


class Laplacian(nn.Module):
    def __init__(self, a=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-torch.abs(x) / self.a)


class SuperGaussian(nn.Module):
    def __init__(self, a=1., b=1., trainable=True):
        super().__init__()
        self.register_parameter('a', nn.Parameter(a * torch.ones(1), trainable))
        self.register_parameter('b', nn.Parameter(b * torch.ones(1), trainable))

    def forward(self, x):
        return torch.exp(-x ** 2 / (2 * self.a ** 2)) ** self.b
