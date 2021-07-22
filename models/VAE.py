# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, TransConvNet, GaussianLayer, GenerativeAE, UpsampledConvNet, FCBlock, DittadiConvNet, DittadiUpsampledConv, VecSCM, VecSCMDecoder
from torch.nn import functional as F
from .utils import act_switch
from abc import ABCMeta, abstractmethod, ABC


class VAEBase(nn.Module, GenerativeAE, ABC):

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.latent_size = params["latent_size"]
        self.gaussian_latent = GaussianLayer(self.latent_size, self.latent_size, params["gaussian_init"])
        self.act = nn.Sigmoid()

    def encode(self, inputs: Tensor):
        codes = self.encoder(inputs)
        z, logvar, mu = self.gaussian_latent(codes)
        return [z, mu, logvar]

    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns latent code (not noise) for given input"""
        return self.encode(inputs)[1]

    def decode(self, noise: Tensor, activate:bool) -> Tensor:
        out = self.decoder(noise)
        if activate: out = self.act(out)
        return out

    def sample_noise_from_prior(self, num_samples:int):
        return self.gaussian_latent.sample_standard(num_samples)

    def sample_noise_from_posterior(self, inputs: Tensor):
        #TODO: change here
        return self.encode(inputs)[0]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate)[0]

    def forward(self, inputs: Tensor, activate:bool=False) -> list:
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
        use_MSE = kwargs.get("use_MSE",True)
        # ELBO = reconstruction term + prior-matching term
        # Note: for both losses we take the average over the batch and sum over the other dimensions
        BCE = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean() #sum over all dimensions except the first one (batch)
        MSE = torch.sum(F.mse_loss(self.act(X_hat), X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        recons_loss = MSE if use_MSE else BCE
        KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1).mean()
        loss = recons_loss + self.beta * KL_weight * KL_loss
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 'KL':KL_loss, 'MSE':MSE, 'BCE':BCE}

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""

        return self.gaussian_latent.prior_range


class VAE(VAEBase):

    def __init__(self, params: dict, dim_in) -> None:
        super().__init__(params)
        self.beta = params["beta"]
        self.dittadi_v = params["dittadi"] # boolean flag determining whether or not to use Dittadi convolutional structure
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params) if not self.dittadi_v \
            else DittadiConvNet(self.latent_size)
        if not self.dittadi_v:
            fc_enc = FCBlock(conv_net.final_dim, [128, 64, self.latent_size], act_switch(params["act"]))
            fc_dec = FCBlock(self.latent_size, [64, 128, conv_net.final_dim], act_switch(params["act"]))

        self.encoder = conv_net if self.dittadi_v else nn.Sequential(conv_net, fc_enc)
        self.decoder_initial_shape = conv_net.final_shape
        deconv_net = UpsampledConvNet(self.decoder_initial_shape, self.dim_in, depth=params["dec_depth"], **params) \
            if not self.dittadi_v else DittadiUpsampledConv(self.latent_size)
        self.decoder = deconv_net if self.dittadi_v else nn.ModuleList([fc_dec, deconv_net])

    def decode(self, noise: Tensor, activate:bool) -> Tensor: #overriding parent class implementation to inser reshaping
        noise = self.decoder[0](noise)
        noise = noise.view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](noise)
        if activate: out = self.act(out)
        return out

class VecVAE(VAEBase):
    """Version of VAE model for vector based (not image based) data"""
    def __init__(self, params: dict, dim_in: int, **kwargs) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super().__init__(params)
        self.dim_in = dim_in[0]
        self.beta = params["beta"]
        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.decoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))

    def decode(self, noise:Tensor, activate:bool):
        # since x is a constant we're always going to get the same output

        output = self.decoder(noise)
        return output