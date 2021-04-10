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

    def __init__(self, dim_in, final_dim, channels_list:list):
        #TODO add selection for non-linearity
        super(ConvNet, self).__init__()
        C, H, W = dim_in 
        self.channels_list = channels_list
        # calculating shape of the image after convolution 
        self.final_shape = channels_list[-1], H // 2**(len(channels_list)), W // 2**(len(channels_list))
        assert all(v>0 for v in self.final_shape), "Input not big enough for the convolutions requested"
        flat_dim = int(np.product(self.final_shape))
        
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

class TransConvBlock(nn.Module):
    """ Implements logic for a transpose convolutional block [transposeConv2d, normalization, activation]"""

    def __init__(self, C_IN, C_OUT, k, s, p):
        #TODO: add check for the norm
        #TODO: add selector for activation 
        super(TransConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(C_IN, out_channels=C_OUT, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(C_OUT),nn.LeakyReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)
        return outputs


class TransConvNet(nn.Module):
    """Implements a Transpose convolutional network (initial reshaping + transpose convolution)"""

    def __init__(self, dim_in, initial_shape, channels_list:list):
        """
            - initial_shape:(C,H,W) -> shape of the input image to the transpose 
                convolution block
        """
        super(TransConvNet, self).__init__()
        flat_dim = int(np.product(initial_shape)) 
        self.fc_reshape = nn.Linear(dim_in, flat_dim)
        self.initial_shape = initial_shape
        C,H,W = initial_shape
        # Stacking the trans-conv layers 
        modules = [] 
        for c in channels_list:
            # transpose conv block with kernel size 2, size 2 and padding 1 
            # doubling the input dimension at every step 
            modules.append(TransConvBlock(C, c, 2, 2, 0))
            C = c
        self.trans_net = nn.Sequential(*modules)


    def forward(self, inputs: Tensor) -> Tensor: 
        reshaped = self.fc_reshape(inputs).view((-1,) + self.initial_shape)
        outputs = self.trans_net(reshaped)
        return outputs