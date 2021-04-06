# Implementation of useful layers 
from torch import nn
from torch import Tensor
import numpy as np


class ConvBlock(nn.Module):
    """ Implements logic for a convolutional block [conv2d, normalization, activation]"""

    def __init__(self, C_IN, C_OUT, k, s, p):
        #TODO: add check for the norm
        #TODO: add selector for activation 
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(C_IN, out_channels=C_OUT, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(C_OUT),nn.LeakyReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)
        return outputs


class ConvNet(nn.Module):
    """ Implements convolutional net with multiple convolutional 
    blocks  + final flattening """

    def __init__(self, dim_in, final_dim, channels_list:list=None):
        #TODO add selection for non-linearity
        super(ConvNet, self).__init__()
        C, H, W = dim_in 
        if channels_list is None: channels_list = [32, 64, 128, 256]
        # calculating shape of the image after convolution 
        final_shape = channels_list[-1], H // 2**(len(channels_list)), W // 2**(len(channels_list))
        print(final_shape)
        assert all(v>0 for v in final_shape), "Input not big enough for the convolutions requested"
        flat_dim = int(np.product(final_shape))
        
        # Stacking the conv layers  
        modules = []
        for c in channels_list:
            # conv block with kernel size 2, size 2 and padding 1 
            # halving the input dimension at every step 
            modules.append(ConvBlock(C, c, 2, 2, 0)) 
            C = c
        modules.append(nn.Flatten())
        modules.append(nn.Linear(flat_dim, final_dim)) #notice no activation here
        self.net = nn.Sequential(*modules)


    def forward(self, inputs: Tensor) -> Tensor: 
        outputs = self.net(inputs)
        return outputs