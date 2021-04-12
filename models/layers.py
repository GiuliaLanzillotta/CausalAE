# Implementation of useful layers
import torch
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

    @staticmethod
    def compute_output_size(H_in, H_w, s, p):
        #TODO also consider dilation
        return int(np.floor((H_in-H_w+2*p)/s + 1))

    @staticmethod
    def same_padding(H_in, H_w, s):
        return int(((s-1)*H_w + H_in -1)/2)


class ConvNet(nn.Module):
    """ Implements convolutional net with multiple convolutional 
    blocks  + final flattening """

    def __init__(self, dim_in, final_dim, channels_list:list, filter_size:int = 2, stride:int=2):
        #TODO add selection for non-linearity
        super(ConvNet, self).__init__()
        C, H, W = dim_in 
        self.channels_list = channels_list
        
        # Stacking the conv layers  
        modules = []
        h = H
        for c in channels_list:
            # conv block with kernel size 2, size 2 and padding 1 
            # halving the input dimension at every step 
            modules.append(ConvBlock(C, c, filter_size, stride, 0))
            h = ConvBlock.compute_output_size(h, filter_size, stride, 0)
            C = c

        # calculating shape of the image after convolution
        self.final_shape = channels_list[-1], h, h # assuming square input image
        assert all(v>0 for v in self.final_shape), "Input not big enough for the convolutions requested"
        flat_dim = int(np.product(self.final_shape))
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


    @staticmethod
    def compute_output_size(H_in, H_w, s, p):
        #TODO also consider dilation
        return int((H_in-1)*s - 2*p + H_w)

class TransConvNet(nn.Module):
    """Implements a Transpose convolutional network (initial reshaping + transpose convolution)"""

    def __init__(self, dim_in, initial_shape, final_shape, channels_list:list, filter_size:int=2, stride:int=2):
        """
            - initial_shape:(C,H,W) -> shape of the input image to the transpose 
                convolution block
        """
        super(TransConvNet, self).__init__()
        self.initial_shape = initial_shape
        self.final_shape = final_shape
        flat_dim = int(np.product(initial_shape))
        self.fc_reshape = nn.Linear(dim_in, flat_dim)
        C,H,W = initial_shape
        # Stacking the trans-conv layers 
        modules = []
        h = H
        for c in channels_list:
            # transpose conv block with kernel size 2, size 2 and padding 1 
            # doubling the input dimension at every step 
            modules.append(TransConvBlock(C, c, filter_size, stride, 0))
            h = TransConvBlock.compute_output_size(h, filter_size, stride, 0)
            C = c
        # reshaping into initial size
        modules.append(nn.Flatten())
        flat_dim_out = int(channels_list[-1] * h**2)
        flat_dim_final = int(np.product(final_shape))
        modules.append(nn.Linear(flat_dim_out, flat_dim_final)) #notice no activation here
        self.trans_net = nn.Sequential(*modules)


    def forward(self, inputs: Tensor) -> Tensor: 
        reshaped = self.fc_reshape(inputs).view((-1,) + self.initial_shape)
        outputs = self.trans_net(reshaped)
        reshaped = outputs.view((-1,) + self.final_shape)
        return reshaped

class GaussianLayer(nn.Module):
    """Stochastic layer with Gaussian distribution.
    This layer parametrizes a Gaussian distribution on the latent space of size
    "latent_size" """
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, inputs):
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(inputs)
        logvar = self.logvar(inputs)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu

    def sample_standard(self, num_samples:int, device) -> Tensor:
        """ Sampling noise from the latent space and generating images
            through the decoder"""
        z = torch.randn(num_samples, self.latent_size).to(device)
        return z

    def interpolate_standard(self):
        #TODO: implement interpolation
        pass

class HybridLayer(nn.Module):
    """Stochastic layer based on Hybrid sampling"""
    def __init__(self, dim, unit_dim, N):
        super(HybridLayer, self).__init__()
        self.dim = dim
        self.unit_dim = unit_dim
        self.N = N
        self.weights = torch.ones(self.N) # unnormalised probability weights for each sample (equivalent to uniform)
        self.prior = None

    def initialise_prior(self, latent_vectors):
        # randomly selecting latent vectors for the prior
        idx = torch.randperm(latent_vectors.size(0))
        if latent_vectors.size(0)<self.N:
            selected_idx = idx[:latent_vectors.size(0)]
        else: selected_idx = idx[:self.N]
        self.prior = latent_vectors[selected_idx]

    def update_prior(self, latent_vectors):
        # TODO: make initialisation incremental (store an incremental pool
        #  over different batches and sample from that)
        pass

    def sample_from_prior(self, input_shape):
        # splitting the prior latent vectors into chunks (one for each noise dimension)
        num_samples = input_shape[0]
        prior_chunks = torch.split(self.prior, self.unit_dim, dim=1)
        # randomising the order of each chunk
        new_vectors = torch.zeros(input_shape, dtype=torch.float).to(self.prior.device)
        for i,chunk in enumerate(prior_chunks):
            # num_samples x N one-hot matrix
            idx = torch.multinomial(self.weights[:self.prior.size(0)], num_samples, replacement=True)
            W = nn.functional.one_hot(idx).float().to(self.prior.device)
            new_vectors[i*self.unit_dim:(i+1)*self.unit_dim] += torch.matmul(W, self.prior)
        return new_vectors

    def forward(self, inputs):
        """Performs hybrid sampling on the latent space"""
        self.initialise_prior(inputs)
        output = self.sample_from_prior(inputs.shape)
        return output


class AdaIN(nn.Module):
    #TODO: implement AdaIN layer (to be used instead of StrTrf in baselines)
    pass

class StrTrf(nn.Module):
    """Implements the structural transform layer. To be used in SAE."""
    def __init__(self, noise_size):
        super().__init__()
        self.fc_mu = nn.Linear(noise_size,1)
        self.sigma = nn.Linear(noise_size,1)

    def forward(self, x, z):
        """ z is the noise obtained via hybrid or parametric sampling.
        x is the set of all causal variables computed so far. Thus this is
        the layer where the causal variables interact with the noise terms to generate
        the children variables."""
        y = self.fc_mu(z) + self.sigma(z)*x
        return y


class UpsampledConv(nn.Module):
    """Implements layer to be used in structural decoder:
    bilinear upsampling + convolution (with activtion) (with no dimensionality reduction)"""
    def __init__(self, upsampling_factor, dim_in, channels:int, filter_size:int=2, stride:int=2):
        super().__init__()
        C, H, W = dim_in
        self.upsampling = nn.Upsample(scale_factor=upsampling_factor, mode="bilinear", align_corners=True)
        padding = ConvBlock.same_padding(H*upsampling_factor, filter_size, stride)
        self.conv_layer = ConvBlock(C, channels, filter_size, stride, padding)
        self.act = nn.ReLU()#TODO: add selection here

    def forward(self, inputs):
        upsmpld = self.upsampling(inputs)
        output = self.act(self.conv_layer(upsmpld))
        return output

class SCMDecoder(nn.Module):
    """ Implements SCM layer (to be used in SAE): given a latent noise vector the
    SCM produces the causal variables by ancestral sampling."""
    def __init__(self, initial_shape, final_shape, latent_size, unit_dim, channels_list:list,
                 filter_size:int=2, stride:int=2, upsampling_factor:int=2):
        super().__init__()
        assert latent_size%unit_dim==0, "The noise vector must be a multiple of the unit dimension"
        # latent size: size of the noise vector in input (obtained by hybrid/parametric sampling)
        self.latent_size = latent_size
        # unit_dim: dimension of one "noise unit"- not necessarily 1
        # the noise will be processed one unit at a time -> the noise vector is
        # split in units during the forward pass
        self.unit_dim = unit_dim
        self.hierarchy_depth = latent_size//unit_dim
        C, H, W = initial_shape # the initial shape refers to the constant input block
        assert len(channels_list)==self.hierarchy_depth, "Specified number of channels not matching heirarchy depth"
        self.conv_modules = []
        self.str_trf_modules = []
        h = H
        for c in channels_list:
            self.str_trf_modules.append(StrTrf(self.unit_dim))
            self.conv_modules.append(UpsampledConv(upsampling_factor, initial_shape, c, filter_size, stride))
            h *= 2
            # reshaping into initial size
        self.flatten = nn.Flatten()
        flat_dim_out = int(channels_list[-1] * h**2)
        flat_dim_final = int(np.product(final_shape))
        self.fc_out = nn.Linear(flat_dim_out, flat_dim_final) #notice no activation here

    def forward(self, x, z):
        start_dim=0
        for sf, conv in zip(self.str_trf_modules, self.conv_modules):
            z_i = z[start_dim:start_dim+self.unit_dim]
            y = sf(x, z_i)
            x = conv(y)
            start_dim+=self.unit_dim
        flattened = self.flatten(x)
        outputs = self.fc_out(flattened)
        reshaped = outputs.view((-1,) + self.final_shape)
        return reshaped







