# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
import numpy as np
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, TransConvNet
from torch.nn import functional as F

class VAE(nn.Module):

    def __init__(self, params:dict, dim_in) -> None:
        super(VAE, self).__init__()
        self.beta = params["beta"]
        self.latent_dim = params["latent_dim"]
        self.dim_in = dim_in # C, H, W
        # Building encoder
        #TODO: add a selection for non-linearity here
        channels_list = params["channels_list"]
        conv_net = ConvNet(dim_in, self.latent_dim, channels_list=channels_list)
        self.conv_net = nn.Sequential(conv_net, nn.ELU())
        self.fc_mu = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_var = nn.Linear(self.latent_dim, self.latent_dim)
        channels_list.reverse()
        self.trans_conv_net = TransConvNet(self.latent_dim, conv_net.final_shape,
                                           channels_list[1:] + [dim_in[0]])

    def encode(self, inputs: Tensor):
        conv_result = self.conv_net(inputs)
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(conv_result)
        log_var = self.fc_var(conv_result)
        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        trans_conv_res = self.trans_conv_net(z)
        return trans_conv_res

    @staticmethod
    def sample_parametric(mu, logvar):
        """Sampling from parametric Gaussian
        Notice that mu and logvar are both multidimensional vectors, and their
        dimension determines the number of samples."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample_standard(self, num_samples:int, device) -> Tensor:
        """ Sampling noise from the latent space and generating images
        through the decoder"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x)[0]

    def forward(self, inputs: Tensor) -> list:
        mu, log_var = self.encode(inputs)
        z = self.sample_parametric(mu, log_var)
        return  [self.decode(z), mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        X_hat = args[0]
        mu = args[1]
        log_var = args[2]
        X = kwargs["X"]
        #  In this context it makes sense to normalise β by latent z size
        # m and input x size n in order to compare its different values
        # across different latent layer sizes and different datasets
        KL_weight = kwargs["KL_weight"] # usually weight = M/N
        # ELBO = reconstruction term + prior-matching term
        recons_loss= F.mse_loss(X_hat, X)
        KL_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + self.beta * KL_weight * KL_loss
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 'KL':KL_loss}
