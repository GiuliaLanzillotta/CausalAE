"""Script for the 'Regularised SAE' model"""
from abc import abstractmethod, ABC

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

from . import FCBlock, SAE, utils, VecSCM, VecSCMDecoder, HybridAE, ConvAE, VecAE
from .utils import act_switch


class RHybridAE(HybridAE, ABC):
    """Regularised version of the HybridSAE: simply adds a regularisation term
    to the reconstruction objective."""
    @abstractmethod
    def __init__(self, params:dict):
        super(RHybridAE, self).__init__(params)

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def forward(self, inputs: Tensor, **kwargs) -> list:
        activate = kwargs.get('activate',False)
        update_prior = kwargs.get('update_prior',False)
        integrate = kwargs.get('integrate',True)
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes, integrate=integrate)
        output = self.decode(codes, activate)
        return  [output, codes]

    def reconstruct(self, x: Tensor, activate:bool) -> Tensor:
        return self.forward(x, activate=activate, update_prior=True, integrate=True)[0]

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        lamda = kwargs.get('lamda_MMD',2.0)
        device = kwargs.get('device','cpu')
        losses = kwargs.get('losses')
        Z = args[1]
        Z_prior = self.hybrid_layer.sample_from_prior(Z.shape)
        L_reg = utils.compute_MMD(Z,Z_prior, kernel=self.kernel_type, **self.params, device=device)
        losses['Regularization_loss'] = L_reg
        losses['loss'] += lamda*L_reg
        return losses


class RSAE(RHybridAE, SAE):
    """Regularised version of the SAE"""
    def __init__(self, params:dict) -> None:
        super(RSAE, self).__init__(params)
        self.kernel_type = params["MMD_kernel"]

    def decode(self, noise:Tensor, activate:bool):
        return SAE.decode(self, noise, activate)

class VecRSAE(RHybridAE):
    """Version of RSAE model for vector based (not image based) data"""
    def __init__(self, params: dict, dim_in: int, full: bool, **kwargs) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super(VecRSAE, self).__init__(params)
        self.dim_in = dim_in[0]
        self.kernel_type = params["MMD_kernel"]

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

class RAE(RHybridAE, ConvAE):
    """Regularised version of the ConvAE"""
    def __init__(self, params:dict) -> None:
        super(RAE, self).__init__(params)
        self.kernel_type = params["MMD_kernel"]

    def decode(self, noise:Tensor, activate:bool):
        return ConvAE.decode(self, noise, activate)

class VecRAE(RHybridAE, VecAE):
    """ Regularised version of the VecAE """
    def __init__(self, params:dict, dim_in, **kwargs) -> None:
        VecAE.__init__(self, params, dim_in)
        self.kernel_type = params["MMD_kernel"]

    def decode(self, noise:Tensor, activate:bool):
        return VecAE.decode(self, noise, activate)