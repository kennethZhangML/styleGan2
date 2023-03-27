import torch 
import torch.nn as nn
import torch.nn.functional as F

class WassersteinLossDiscriminator(nn.Module):

    def __init__(self, f_real : torch.Tensor, f_fake : torch.Tensor):

        super().__init__()

        self.f_real = f_real
        self.f_fake = f_fake
    
    def forward(self):
        return F.relu(1 - self.f_real).mean(), F.relu(1 + self.f_fake).mean()

class WassersteinLossGenerator(nn.Module):

    def __init__(self, f_fake : torch.Tensor):

        super().__init__()

        self.f_fake = f_fake
    
    def forward(self):
        return -self.f_fake.mean()

    

