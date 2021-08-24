"""Module containing implementation of standard (Visual and Vector) autoencoder"""
import torch
from torch import Tensor
# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from torch import nn

from . import ConvNet, UpsampledConvNet, FCBlock, HybridAE, VecSCM, Xnet
from .utils import act_switch


class ConvAE(HybridAE):

    def __init__(self, params: dict) -> None:
        super(ConvAE, self).__init__(params)
        dim_in = params['dim_in']
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params)
        conv_fin = FCBlock(conv_net.final_dim, [256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, conv_fin) # returns vector of latent_dim size
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = conv_net.final_shape
        dec_init = FCBlock(self.latent_size, [128, 256, conv_net.final_dim], act_switch(params.get("act")))
        deconv_net = UpsampledConvNet(self.decoder_initial_shape, self.dim_in, depth=params["dec_depth"], **params)
        self.decoder = nn.ModuleList([dec_init, deconv_net])

    def decode(self, noise:Tensor, activate:bool):
        codes = self.decoder[0](noise)
        codes = codes.view((-1, )+self.decoder_initial_shape) # reshaping into image format
        output = self.decoder[1](codes)
        if activate: output = self.act(output)
        return output

    def add_regularisation_terms(self, *args, **kwargs):
        losses = kwargs.get('losses')
        return losses

class XAE(ConvAE, Xnet):

    def __init__(self, params: dict) -> None:
        super(XAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool):
        z = self.causal_block(noise, self.tau)
        z_init = self.decoder[0](z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        x = self.decoder[1](z_init)
        if activate: x = self.act(x)
        return x

    def add_regularisation_terms(self, *args, **kwargs):
        return Xnet.add_regularisation_terms(self, *args, **kwargs)


class VecAE(HybridAE):
    """Simply an AE made of fully connected layers"""
    def __init__(self, params: dict, dim_in: int, **kwargs) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super().__init__(params)
        self.dim_in = dim_in[0]
        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.decoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))

    def decode(self, noise:Tensor, activate:bool):
        output = self.decoder(noise)
        return output