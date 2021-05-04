# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, TransConvNet, GaussianLayer, GenerativeAE, UpsampledConvNet
from torch.nn import functional as F

class VAE(nn.Module, GenerativeAE):

    def __init__(self, params:dict, dim_in) -> None:
        super(VAE, self).__init__()
        self.beta = params["beta"]
        self.latent_size = params["latent_size"]
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, self.latent_size, depth=params["enc_depth"], **params)
        self.conv_net = conv_net
        self.gaussian_latent = GaussianLayer(self.latent_size, self.latent_size)
        self.upsmpld_conv_net = UpsampledConvNet((self.latent_size, 1, 1), self.dim_in,
                                                 depth=params["dec_depth"], **params)
        self.act = nn.Sigmoid()


    def encode(self, inputs: Tensor):
        conv_result = self.conv_net(inputs)
        z, logvar, mu = self.gaussian_latent(conv_result)
        return [z, mu, logvar]

    def decode(self, noise: Tensor, activate:bool) -> Tensor:
        upsmpld_res = self.upsmpld_conv_net(noise)
        if activate: upsmpld_res = self.act(upsmpld_res)
        return upsmpld_res

    def sample_noise_from_prior(self, num_samples:int):
        return self.gaussian_latent.sample_standard(num_samples)

    def sample_noise_from_posterior(self, inputs: Tensor):
        return self.encode(inputs)[0]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate)[0]

    def forward(self, inputs: Tensor, activate:bool=False) -> list:
        inputs = inputs.view((-1, )+self.dim_in)
        z, mu, logvar = self.encode(inputs)
        return  [self.decode(z, activate), mu, logvar]

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
        recons_loss= F.binary_cross_entropy_with_logits(X_hat, X, reduction="mean")
        KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = recons_loss + self.beta * KL_weight * KL_loss
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 'KL':KL_loss}
