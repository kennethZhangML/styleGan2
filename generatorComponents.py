import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn

from miscComponents import DownSample, BatchCreator, Conv2dModulate, ToRGB, Upsample
from equalizerComponents import EqualizedConv2d, EqualizedFCLinear


class Generator(nn.Module):

    def __init__(self, log_res : int, d_lat : int, n_features : int = 32, max_features : int = 512):

        super().__init__()

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_res - 2, -1, -1)]
        self.n_blocks = len(features)

        self.init_constant = nn.Parameter(torch.randn((1, features[0], 4, 4)))
        self.style_block = StyleBlock(d_lat, features[0], features[0])
        self.to_rgb = ToRGB(d_lat, features[0])

        blocks = [GeneratorBlock(d_lat, features[i - 1], features[i]) for i in range(1, self.n_blocks)]
        self.blocks = nn.ModuleList(blocks)

        self.up_sample = Upsample()

    def forward(self, weight : torch.Tensor, input_noise : List[Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]):
        batch_size = weight.shape[1]

        X = self.init_constant.expand(batch_size, -1, -1, -1)
        X = self.style_block(X, weight[0], input_noise[0][1])

        rgb = self.to_rgb(X, weight[0])

        for i in range(1, self.n_blocks):
            X = self.up_sample(X)
            X, rgb_new = self.blocks[i - 1](X, weight[i], input_noise[i])
            rgb = self.up_sample(rgb) + rgb_new
        return rgb


class DiscriminatorBlock(nn.Module):

    def __init__(self, in_features : int, out_features : int):
        super().__init__()

        self.residual = nn.SEquential(DownSample(),
                                      EqualizedConv2d(in_features, out_features, kernel_size = 1))
        
        self.block_residual = nn.Sequential(
            nn.EqualizedConv2d(in_features, in_features, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2, True),
            EqualizedConv2d(in_features, out_features, kernel_size = 3, padding = 1),
            nn.LeakyReLU(0.2, True)
        )

        self.down_sample = DownSample()
        self.scale = 1 / math.sqrt(2)
    
    def forward(self, X):

        residual =  self.residual(X)

        X = self.block(X)
        X = self.down_sample(X)
        return (X + residual) * self.scale


class Discriminator(nn.Module):

    def __init__(self, log_resolution : int, n_features: int = 64, 
                 max_features : int = 512):
        
        super().__init__()

        self.from_rgb = nn.Sequential(
            EqualizedConv2d(3, n_features, 1),
            nn.LeakyReLU(0.2, True)
        )

        features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]

        n_blocks = len(features) - 1
        blocks = [DiscriminatorBlock(features[i], features[i + 1]) for i in range(n_blocks)]
        self.blocks = nn.Sequential(*blocks)

        self.std_dev = BatchCreator()
        final_features = features[-1] + 1

        self.conv = EqualizedConv2d(final_features, final_features, 3)
        self.final = EqualizedFCLinear(2 * 2 * final_features, 1)

    def forward(self, X : torch.Tensor):
        X = X - 0.5
        X = self.from_rgb(X)

        X = self.blocks(X)
        X = self.std_dev(X)
        X = self.conv(X)
        X = X.reshape(X.shape[0], -1)
        return self.final(X)
    
class StyleBlock(nn.Module):

    def __init__(self, d_latent : int, in_features : int, out_features : int):

        super().__init__()

        self.to_style = EqualizedFCLinear(d_latent, in_features, bias = 1.0)
        self.conv = Conv2dModulate(in_features, out_features, kernel_size = 3)
        self.scale_noise = nn.Parameter(torch.zeros(1))
        self.bias = nn.Parameter(torch.zeros(out_features))

        self.activation = nn.LeakyReLU(0.2, True)

    def forward(self, X : torch.Tensor, weight : torch.Tensor, noise : Optional[torch.Tensor]):

        s = self.to_style(weight)
        weight = self.conv(weight, s)

        if noise is not None:
            X = X + self.scale_noise[None, :, None, None] * noise
        return self.activation(X + self.bias[None, :, None, None])

class GeneratorBlock(nn.Module):

    def __init__(self, d_latent : int, in_features : int, out_features: int):
        super().__init__()

        self.style_block1 = StyleBlock(d_latent, in_features, out_features)
        self.style_block2 = StyleBlock(d_latent, out_features, out_features)

        self.to_rgb = ToRGB(d_latent, out_features)

    def forward(self, X : torch.Tensor, weight : torch.Tensor, noise : Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]):

        X = self.style_block1(X, weight, noise[0])
        X = self.style_block2(X, weight, noise[1])
        rgb = self.to_rgb(X, weight)
        return X, rgb