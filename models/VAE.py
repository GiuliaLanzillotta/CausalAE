# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from abc import ABC

import torch
from torch import Tensor
from torch import nn
from models import utils, Xnet
from torch.nn import functional as F

from . import ConvNet, GaussianLayer, GenerativeAE, UpsampledConvNet, FCBlock, DittadiConvNet, DittadiUpsampledConv, \
    VecSCM
from .utils import act_switch, KL_multiple_univariate_gaussians


class VAEBase(GenerativeAE, nn.Module, ABC):

    def __init__(self, params):
        super(VAEBase, self).__init__(params)
        self.params = params
        self.latent_size = params["latent_size"]
        self.unit_dim = params.get('unit_dim',1)
        self.gaussian_latent = GaussianLayer(self.latent_size, self.latent_size, params["gaussian_init"])

    def encode(self, inputs: Tensor, **kwargs):
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

    def sample_noise_from_prior(self, num_samples: int, **kwargs):
        return self.gaussian_latent.sample_standard(num_samples)

    def sample_noise_from_posterior(self, inputs: Tensor):
        #TODO: change here --- why?
        return self.encode(inputs)[0]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate=activate)[0]

    def forward(self, inputs: Tensor, **kwargs) -> list:
        activate= kwargs.get('activate',False)
        z, mu, logvar = self.encode(inputs)
        return  [self.decode(z, activate), mu, logvar]

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        mu = args[1]
        log_var = args[2]
        X = kwargs["X"]
        device = kwargs.get('device')
        losses = kwargs.get('losses')
        """In this context it makes sense to normalise β by latent z size
        m and input x size n in order to compare its different values
        across different latent layer sizes and different datasets"""
        KL_weight = kwargs["KL_weight"] # usually weight = M/N
        KL_loss = KL_multiple_univariate_gaussians(mu, torch.zeros_like(mu).to(device),
                                                   log_var, torch.zeros_like(log_var).to(device),
                                                   reduce=True)
        losses['KL'] = KL_loss
        losses['loss'] += self.beta * KL_weight * KL_loss
        return losses

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""

        return self.gaussian_latent.prior_range

class VAE(VAEBase):

    def __init__(self, params: dict) -> None:
        super(VAE, self).__init__(params)
        self.beta = params["beta"]
        self.dittadi_v = params.get('dittadi',False) # boolean flag determining whether or not to use Dittadi convolutional structure
        dim_in = params['dim_in']
        self.dim_in = dim_in # C, H, W        # Building encoder
        conv_net = ConvNet(depth=params["enc_depth"], **params) if not self.dittadi_v \
            else DittadiConvNet(self.latent_size)
        if not self.dittadi_v:
            fc_enc = FCBlock(conv_net.final_dim, [128, 64, self.latent_size], act_switch(params["act"]))
            fc_dec = FCBlock(self.latent_size, [64, 128, conv_net.final_dim], act_switch(params["act"]))

        self.encoder = conv_net if self.dittadi_v else nn.Sequential(conv_net, fc_enc)
        self.decoder_initial_shape = conv_net.final_shape
        deconv_net = UpsampledConvNet(self.decoder_initial_shape, final_shape=self.dim_in, depth=params["dec_depth"], **params) \
            if not self.dittadi_v else DittadiUpsampledConv(self.latent_size)
        self.decoder = deconv_net if self.dittadi_v else nn.ModuleList([fc_dec, deconv_net])

    def decode(self, noise: Tensor, activate:bool) -> Tensor: #overriding parent class implementation to inser reshaping
        noise = self.decoder[0](noise)
        noise = noise.view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](noise)
        if activate: out = self.act(out)
        return out

class XVAE(VAE, Xnet):
    """ Explicit latent block + VAE """
    def __init__(self, params: dict) -> None:
        super(XVAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool) -> Tensor: #overriding parent class implementation to inser reshaping
        Z = self.causal_block(noise, masks_temperature=self.tau)
        out_init = self.decoder[0](Z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](out_init)
        if activate: out = self.act(out)
        return out

    def add_regularisation_terms(self, *args, **kwargs):
        losses = VAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = Xnet.add_regularisation_terms(self, *args, **kwargs)
        return losses

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