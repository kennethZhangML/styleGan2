import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from equalizerComponents import EqualizedWeight, EqualizedFCLinear

class PenaltyPathLen(nn.Module):

    def __init__(self, beta : float):
        super().__init__()

        self.beta = beta
        self.steps = nn.Parameter(torch.tensor(0.), requires_grad = False)
        self.exp_sum = nn.Parameter(torch.tensor(0.), requires_grad = False)
    
    def forward(self, weights : torch.Tensor, X : torch.Tensor):
        device = X.device
        image_size = X.shape[2] * X.shape[3]
        y = torch.randn(X.shape, device = device)
        output = (X * y).sum() / math.sqrt(image_size)

        grads, *_ = torch.autograd.grad(outputs = output,
                            inputs = weights, grad_outputs = torch.ones(output.shape, device = device),
                            create_graph = True)
        
        norms = (grads ** 2).sum(dim = 2).mean(dim = 1).sqrt()

        if self.steps > 0:
            a = self.exp_sum / (1 - self.beta ** self.steps)
            loss = torch.mean((norms - a) ** 2)

        else: 
            loss = norms.new_tensor(0)
        
        mean = norms.mean().detach()
        self.exp_sum.mul_(self.beta).add_(mean, alpha = 1 - self.beta)
        self.steps.add_(1.)

        return loss

class GradientPenalisation(nn.Module):

    def __init__(self, X : torch.Tensor, d : torch.Tensor):

        super().__init__()
        
        self.batch_size = X.shape[0]
        grads, *_ = torch.autograd.grad(outputs = d, inputs = X, 
                            grad_outputs = d.new_ones(d.shape),
                            create_graph = True)
        
        gradients = gradients.reshape(self.batch_size, -1)
        norms = gradients.norm(2, dim = -1)
        return torch.mean(norms ** 2)
    
class SmoothingFunction(nn.Module):

    def __init__(self):

        super().__init__()

        self.kernel = [[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]]
        
        self.kernel = torch.tensor([[self.kernel]], dtype = torch.float)
        self.kernel /= self.kernel.sum()

        self.kernel = nn.Parameter(self.kernel, requires_grad = False)
        self.pad = nn.ReplicationPad2d(1)
    
class Upsample(nn.Module):

    def __init__(self):
        super().__init__()

        self.up_sample = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = False)
        self.smooth = SmoothingFunction()

    def forward(self, X : torch.Tensor):
        return self.smooth(self.up_sample(X))

class DownSample(nn.Module):

    def __init__(self):
        super().__init__()

        self.smooth = SmoothingFunction()
    
    def forward(self, X : torch.Tensor):
        X = self.smooth(X)
        return F.interpolate(X, (X.shape[2] // 2, X.shape[3] // 2), mode = 'bilinear', align_corners = False)

class BatchCreator(nn.Module):

    def __init__(self, group_size : int = 4):

        super().__init__()
        self.group_size = group_size
    
    def forward(self, X : torch.Tensor):

        assert X.shape[0] % self.group_size == 0

        grouped = X.view(self.group_size, -1)
        std = torch.sqrt(grouped.var(dim = 0) + 1e-8)
        std = std.mean().view(1, 1, 1, 1)

        b, _, h, w = X.shape
        std = std.expand(b, -1, h, w)
        return torch.cat([X, std], dim = 1)
    
class Conv2dModulate(nn.Module):

    def __init__(self, in_features : int, out_features : int, kernel_size : int,
                 demodulate : float = True, epsilon : float = 1e-8):
        
        super().__init__()

        self.out_features = out_features
        self.demodulate = demodulate
        self.padding = (kernel_size - 1) // 2

        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.epsilon = epsilon
    
    def forward(self, X : torch.Tensor, s : torch.Tensor):
        base, _, height, width = X.shape
        s = s[:, None, :, None, None]

        weights = self.weight()[None, :, :, :, :]
        weights = weights * s

        if self.demodulate:
            inv_sig = torch.rsqrt((weights ** 2).sum(dim = (1, 2, 3, 4), keepdim = True) + self.eps)
            weights = weights + inv_sig
        
        X = X.reshape(1, -1, height, width)
        _, _, *ws = weights.shape
        weights = weights.reshape(base * self.out_features, *ws)
        
        X = F.conv2d(X, weights, padding = self.padding, groups = base)
        return X.reshape(-1, self.out_features, height, width)
    
class ToRGB(nn.Module):

    def __init__(self, d_latent : int, features : int):

        super().__init__()

        self.to_style = EqualizedFCLinear(d_latent, features, bias = 1.0)

        self.conv = Conv2dModulate(features, 3, kernel_size = 1, demodulate = False)
        self.bias = nn.Paramter(torch.zeros(3))
        self.activation = nn.LeakyReLU(0.2, True)
    
    def forward(self, X : torch.Tensor, weight : torch.Tensor):
        style = self.to_style(weight)
        X = self.conv(X, style)
        return self.activation(X + self.bias[None, :, None, None])
