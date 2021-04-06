# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl 
import numpy as np
from torch import nn
import torch
from . import ConvNet, TransConvNet

class VAE(nn.Module):

    def __init__(self, dim_in, latent_dim, channels_list:list=None) -> None:
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        # Building encoder
        #TODO: add a selection for non-linearity here
        if channels_list is None: channels_list = [32, 64, 128, 256]
        conv_net = ConvNet(dim_in, latent_dim,channels_list=channels_list)
        self.conv_net = nn.Sequential(conv_net, nn.ELU())
        self.fc_mu = nn.Linear(latent_dim, latent_dim)
        self.fc_var = nn.Linear(latent_dim, latent_dim)
        channels_list.reverse()
        self.trans_conv_net = TransConvNet(latent_dim, conv_net.final_shape, channels_list[1:] + [dim_in[0]])

    def encode(self, input: Tensor):
        conv_result = self.conv_net(input)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(conv_result)
        log_var = self.fc_var(conv_result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        trans_conv_res = self.trans_conv_net(z)
        return trans_conv_res

    def sample_parametric(self, mu, logvar):
        """Sampling from parametric Gaussian
        Notice that mu and logvar are both multidimensional vectors, and their 
        dimension determines the number of samples."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_standard(self, batch_size:int) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def forward(self, input: Tensor) -> Tensor:
        mu, log_var = self.encode(input)
        z = self.sample_parametric(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass