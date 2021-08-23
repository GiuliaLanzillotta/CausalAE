""" Implements interface for generative autoencoders """
from abc import ABCMeta, abstractmethod, ABC
from typing import List

import torch
from torch import Tensor
from torch import nn
from torch import Tensor
from torch.distributions import Uniform
from . import utils
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, VecSCMDecoder, VecSCM
from torch.nn import functional as F
from abc import ABCMeta, abstractmethod, ABC


class GenerativeAE(ABC):

    latent_size = None
    act = nn.Sigmoid()

    @abstractmethod
    def encode(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns all the encoder's output (noise and code included) for given input"""
        raise NotImplementedError

    @abstractmethod
    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns latent code (not noise) for given input"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool) -> Tensor:
        """ returns generated sample given noise"""
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_prior(self, num_samples:int, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_posterior(self, inputs:Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs:Tensor, activate:bool) -> Tensor:
        """ generates output from input"""
        raise NotImplementedError

    @abstractmethod
    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        #TODO: change to aggregate posterior range

    @abstractmethod
    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        raise NotImplementedError

    def loss_function(self, *args, **kwargs):
        """General loss function for standard AE
        Computes reconstruction error internally and
        retrieves regularization terms externally"""

        use_MSE = kwargs.get('use_MSE',True)
        losses = {}

        X_hat = args[0]
        X = kwargs.get('X')
        MSE,BCE = utils.pixel_losses(X,X_hat, act=self.act)
        L_rec = MSE if use_MSE else BCE

        losses['Reconstruction_loss'] = L_rec
        losses['loss'] = L_rec

        losses = self.add_regularisation_terms(*args, losses=losses, **kwargs)

        return losses


class HybridAE(GenerativeAE, nn.Module, ABC):
    """Generalisation of all the generative autoencoder models that adopt hybrid layers
    as stochastic layers"""
    def __init__(self, params: dict):
        nn.Module.__init__(self)
        self.params = params
        self.latent_size = params["latent_size"]
        self.unit_dim = params["unit_dim"]
        self.N = params["latent_vecs"] # number of latent vectors to store for hybrid sampling
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_size, self.unit_dim, self.N)

    def encode(self, inputs: Tensor, **kwargs):
        codes = self.encoder(inputs)
        _update_prior = kwargs.get("update_prior", False)
        _integrate = kwargs.get("integrate", False)
        if _update_prior: self.hybrid_layer.update_prior(codes, integrate=_integrate)
        return codes

    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns latent code (not noise) for given input"""
        codes =  self.encode(inputs)
        _update_prior = kwargs.get("update_prior", False)
        _integrate = kwargs.get("integrate", False)
        if _update_prior: self.hybrid_layer.update_prior(codes, integrate=_integrate)
        return codes

    def sample_hybrid(self, num_samples:int=None, inputs:Tensor=None):
        """ Samples from hybrid distribution
        If inputs is None the distribution will be based on the stored codes.
        Else if inputs is given the distribution will be based on the computed codes."""
        if inputs is None:
            return self.hybrid_layer.sample_from_prior((num_samples,))
        codes = self.encode(inputs, update_prior=True, integrate=True)
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    def sample_noise_from_prior(self, num_samples: int, **kwargs):
        """Equivalent to total hybridisation: every point is hybridised at the maximum level"""
        uniform = kwargs.get('uniform',False)
        if not uniform: return self.hybrid_layer.sample_from_prior((num_samples,))
        # compute extremes of uniform distribution over the latent space using the stored codes
        lows = torch.min(self.hybrid_layer.prior, dim=0).values
        highs = torch.min(self.hybrid_layer.prior, dim=0).values
        uniform = Uniform(lows, highs)
        return uniform.sample([num_samples])

    def sample_noise_from_posterior(self, inputs: Tensor):
        """Posterior distribution of a standard autoencoder is given by the set of all samples"""
        self.encode(inputs, update_prior=True, integrate=True)
        return self.hybrid_layer.prior

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate, update_prior=True, integrate=True)[0]

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False, integrate=True) -> List[Tensor]:
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes, integrate=integrate)
        output = self.decode(codes, activate)
        return  [output]

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        return self.hybrid_layer.prior_range


class Xnet():
    """AE with explicit causal block"""
    causal_block = None
    sparsity_on = None

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        losses = kwargs.get('losses')
        lamda = kwargs.get('sparsity_lamda')
        if self.sparsity_on:
            sparsity_penalty = self.causal_block.masks_sparsity_penalty()
            losses['sparsity_penalty'] = sparsity_penalty
            losses['loss'] += lamda*sparsity_penalty
        return losses
