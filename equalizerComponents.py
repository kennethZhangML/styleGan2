import math
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn


class EqualizedWeight(nn.Module):

    def __init__(self, shape : list[int]):

        super().__init__()

        self.init_constant = 1 / math.sqrt(np.prod(shape[1:]))
        self.weight = nn.Paramter(torch.randn(shape))

    def forward(self):
        return self.weight * self.init_constant
    
class EqualizedFCLinear(nn.Module):

    def __init__(self, in_features : int, out_features : int, bias : float = 0.):
        super().__init__()

        self.weights = EqualizedWeight([out_features, in_features])
        self.bias = nn.Parameter(torch.ones(out_features) * bias)
    
    def forward(self, X : torch.Tensor):
        return F.Linear(X, self.weights(), bias = self.bias)

class EqualizedConv2d(nn.Module):

    def __init__(self, in_features : int, out_features : int, 
                 kernel_size : int, padding : int = 0):
        
        super().__init__()

        self.padding = padding
        self.weight = EqualizedWeight([out_features, in_features, kernel_size, kernel_size])
        self.bias = nn.Parameter(torch.ones(out_features))

