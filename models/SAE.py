# Code for SAE (no attention in the encoder)
# Paper: https://arxiv.org/abs/2006.07796
import torch
from torch import Tensor
from torch import nn

from . import ConvNet, SCMDecoder, FCBlock, VecSCMDecoder, VecSCM, HybridAE, act_switch


class SAE(HybridAE):

    def __init__(self, params: dict, dim_in) -> None:
        HybridAE.__init__(self, params)
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params)
        fc_net = FCBlock(conv_net.final_dim, [256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, fc_net) # returns vector of latent_dim size
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = conv_net.final_shape
        self.dec_init = FCBlock(self.latent_size, [128, 256, conv_net.final_dim], act_switch(params.get("act")))
        self.scm = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)

    def decode(self, noise:Tensor, activate:bool):
        # feeding a constant signal into the decoder
        # the output will be built on top of this constant trough the StrTrf layers
        x = torch.ones(size = noise.shape).to(noise.device) # batch x latent
        # passing x through the linear layers does not make much sense:
        # since x is a constant we're always going to get the same output
        #TODO: this is useless delete the dec_init
        x = self.dec_init(x).view((-1, )+self.decoder_initial_shape) # batch x 512
        output = self.scm(x, noise)
        if activate: output = self.act(output)
        return output



class XSAE(HybridAE):

    def __init__(self, params: dict, dim_in) -> None:
        HybridAE.__init__(self, params)
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params)
        fc_net = FCBlock(conv_net.final_dim, [256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, fc_net) # returns vector of latent_dim size
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = conv_net.final_shape
        # 1. vecSCM N -> Z (causal block)
        # - mapping the latent code to the new causal space with an SCM-like structure
        self.caual_block = VecSCM(self.latent_size, **params)
        # 2. SCM Z + constant -> X (decoder)
        # - keeping the SCM like structure in the decoder
        self.decoder = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)

    def decode(self, noise:Tensor, activate:bool):
        z = self.caual_block(noise)
        # feeding a constant signal into the decoder
        # the output will be built on top of this constant trough the StrTrf layers
        x = torch.ones((noise.shape[0],)+self.decoder_initial_shape).to(noise.device) # batch x latent
        output = self.decoder(x, z)
        if activate: output = self.act(output)
        return output


class VecSAE(HybridAE):
    """Version of SAE model for vector based (not image based) data"""
    def __init__(self, params: dict, dim_in: int, full: bool, **kwargs) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super().__init__(params)
        self.dim_in = dim_in[0]
        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.full = full
        if not full:
            scm = VecSCM(self.latent_size, self.unit_dim, act=params.get("act"))
            reverse_encoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))
            self.decoder = nn.Sequential(scm, reverse_encoder)
        else: self.decoder = VecSCMDecoder(self.latent_size, self.unit_dim, list(reversed(layers)), act=params.get("act"))

    def decode(self, noise:Tensor, activate:bool):
        # since x is a constant we're always going to get the same output
        if not self.full:
            output = self.decoder(noise)
        else:
            x = torch.ones_like(noise).to(noise.device)
            output = self.decoder(x, noise)
        if activate: output = self.act(output)
        return output

