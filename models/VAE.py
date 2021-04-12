# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, TransConvNet, GaussianLayer
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
        conv_net = ConvNet(dim_in, self.latent_dim, channels_list=channels_list,
                           filter_size=params["filter_size"], stride=params["stride"])
        self.conv_net = conv_net
        self.gaussian_latent = GaussianLayer(self.latent_dim, self.latent_dim)
        channels_list.reverse()
        self.trans_conv_net = nn.Sequential(TransConvNet(self.latent_dim, conv_net.final_shape, dim_in,
                                                         channels_list[1:] + [dim_in[0]],
                                                         filter_size=params["filter_size"],
                                                         stride=params["stride"]),
                                            nn.Sigmoid())

    def encode(self, inputs: Tensor):
        conv_result = self.conv_net(inputs)
        z, logvar, mu = self.gaussian_latent(conv_result)
        return [z, mu, logvar]

    def decode(self, z: Tensor) -> Tensor:
        trans_conv_res = self.trans_conv_net(z)
        return trans_conv_res

    def generate_standard(self, num_samples:int, device) -> Tensor:
        """ Sampling noise from the latent space and generating images
        through the decoder"""
        z = self.gaussian_latent.sample_standard(num_samples, device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x)[0]

    def forward(self, inputs: Tensor) -> list:
        z, mu, logvar = self.encode(inputs)
        return  [self.decode(z), mu, logvar]

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
        recons_loss= F.binary_cross_entropy(X_hat, X, reduction="sum")
        KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + self.beta * KL_weight * KL_loss
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 'KL':KL_loss}
