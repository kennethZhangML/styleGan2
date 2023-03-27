import math
from typing import Tuple, Optional, List, Iterator
import numpy as np

import pathlib
from pathlib import Path
from PIL import Image

import torch 
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data 

from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import transforms

from generatorComponents import *
from miscComponents import *
from equalizerComponents import *
from lossComponenets import *


class MasterDataset(torch.utils.data.Dataset):

    def __init__(self, path : str, image_size : int):
        super().__init__()

        self.paths = [path for path in Path(path).glob(f'**/*.jpg')]

        self.transform = torchvision.transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)
    
    def __getitem(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        return self.transform(img)
                
    
class Configurations(nn.Module):

    def __init__(self):

        super().__init__()

    '''
    Define the following configuration settings:

    - device 
    - discriminator class used
    - generator class
    - mapping class used
    
    Loss Functions:
    - discriminator loss function: (wasserstein loss)
    - generator loss

    Optimizers:
    - discriminator optimizer: (Adam optimizer)
    - Generator optimizer: (Adam optimizer)
    - mapping network optimizer: (Adam optimizer)

    Misc Penalties:
    - Gradient Penalty: GradientPenalty()
    - penalty coefficient : float = 10.0

    Path Length Penaly : Path Length Penalty : Any

    Dataloader: Iterator

    batch_size = 32
    d_latent = 512
    image_size = 32
    learning_rate = 1e-3
    mapping_network_learning_rate = 1e-5
    gradient_steps = 1
    adam_betas = (0.0, 0.99)
    style_mixing_probabilities = 0.9
    training_steps = 150_000

    n_generator_blocks: int 

    lazy_gradient_penatly_interval = 4
    lazy_path_penalty_interval = 32
    lazy_path_penalty_after = 5000

    log_generated_interval = 500
    save_chckpt_interval = 2000

    mode = ModeState
    log_layer_outputs = False

    dataset_path = str    
    '''

    '''
    Model-specific parameters
    '''
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    discriminator = Discriminator()
    generator = Generator()
    mapping_network = MappingBlock()

    discriminator_loss = WassersteinLossDiscriminator()
    generator_loss = WassersteinLossGenerator()

    discriminator_optim = torch.optim.Adam
    generator_optim = torch.optim.Adam
    mapping_network_optim = torch.optim.Adam

    gradient_penalty = GradientPenalisation()
    gradient_penalty_coef = 10.0

    path_length_penalty = PenaltyPathLen()

    loader = Iterator

    '''
    Training Loop Specific parameters
    '''

    batch_size = 32
    d_latent = 512
    image_size = 32
    mapping_network_layers = 8
    learning_rate = 1e-3
    mapping_network_learning_rate = 1e-5
    gradient_accumulate_steps = 1
    adam_betas = (0.0, 0.99)
    style_mixing_probs = 0.9

    training_steps = 150000
    n_gen_blocks : int

    lazy_gradient_penalty_interval = 4
    lazy_path_penalty_interval = 32
    lazy_path_penalty_after = 5000

    log_generated_interval : int = 500
    save_chckpt = 2000

    log_layer_outputs: bool = False

    dataset_path = str






    





