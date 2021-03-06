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
    
    def __init__(self, params):
        super(GenerativeAE, self).__init__()

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
    def sample_noise_from_posterior(self, inputs:Tensor, device:str) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def generate(self, num_samples:int, activate:bool, **kwargs) -> Tensor:
        """Generates new samples by sampling noise from prior and decoding the noise"""
        raise NotImplementedError

    @abstractmethod
    def reconstruct(self, inputs:Tensor, activate:bool) -> Tensor:
        """ generates output from input"""
        raise NotImplementedError

    def get_representation(self, inputs:Tensor, **kwargs):
        """ returns a representation vector for the given inputs"""
        return self.encode_mu(inputs, **kwargs)

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

    @abstractmethod
    def forward(self,  inputs: Tensor, **kwargs)-> List[Tensor]:
        """Implements forward pass through the model"""
        raise NotImplementedError

    def get_causal_variables(self, noises:Tensor, **kwargs):
        """Generic function to allow diversity between models in definition/computation of causal variables
        given the noises (i.e. the output of 'encode')"""
        return noises

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
        super(HybridAE, self).__init__(params)
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
        if _update_prior:
            with torch.no_grad(): self.hybrid_layer.update_prior(codes, integrate=_integrate)
        return codes

    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns latent code (not noise) for given input"""
        codes =  self.encode(inputs, **kwargs)
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

    def sample_noise_from_prior(self, num_samples: int, prior_mode='hybrid', **kwargs):
        """Equivalent to total hybridisation: every point is hybridised at the maximum level
        3 modes supported: posterior/hybrid/uniform"""
        device= kwargs['device']

        with torch.no_grad():
            if prior_mode=='posterior':
                # shuffling and selecting
                idx = torch.randperm(self.hybrid_layer.prior.shape[0], device=device) #shuffling indices
                return torch.index_select(self.hybrid_layer.prior, 0, idx[:num_samples]).detach()
            if prior_mode=='hybrid': return self.hybrid_layer.sample_from_prior((num_samples,)).to(device)
            if prior_mode=='uniform': return self.hybrid_layer.sample_uniformly_in_support(num_samples).to(device)
        raise NotImplementedError('Requested sampling mode not implemented')

    def sample_noise_from_posterior(self, inputs: Tensor, device:str):
        """Posterior distribution of a standard autoencoder is given by the set of all samples"""
        self.encode(inputs, update_prior=True, integrate=True)
        return self.hybrid_layer.prior.to(device)

    def generate(self, num_samples:int, activate: bool, **kwargs) -> Tensor:
        device = kwargs.get('device','cpu')
        noise = self.sample_noise_from_prior(num_samples, **kwargs).to(device)
        outputs = self.decode(noise, activate=activate)
        return outputs

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def reconstruct(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x,activate=activate, update_prior=True, integrate=True)[0]

    def forward(self, inputs: Tensor, **kwargs)-> List[Tensor]:
        activate = kwargs.get('activate',False)
        update_prior = kwargs.get('update_prior',False)
        integrate = kwargs.get('integrate',True)
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes, integrate=integrate)
        output = self.decode(codes, activate)
        return  [output]

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        return self.hybrid_layer.prior_range


class Xnet(GenerativeAE, ABC):
    """AE with explicit causal block"""

    def __init__(self, params):
        super(Xnet, self).__init__(params)
        self.sparsity_on = params.get("sparsity",False)
        self.xunit_dim = params.get("xunit_dim",1)
        params['latent_size_prime'] = params['latent_size']*self.xunit_dim
        self.causal_block = VecSCM(use_masking = True, **params)
        self.tau = 1.0

    @abstractmethod
    def decode_from_X(self, x, *args, **kwargs):
        """Implementing the decoding process starting directly from X.
        Useful to inspect the result of an intervention on the causal variables."""
        raise NotImplementedError

    def intervene_on_X(self, dims:List[int], noises:Tensor, values:Tensor):
        """Applies interventions on X at dimensions 'dims' by assigning them the respective 'values'
        Note: noises and values have to be of the same length - for each sample in noises, values must contain a
        corresponding intervention"""
        X = self.causal_block.forward_intervention(noises, dims, values, self.tau)
        return X

    def get_causal_variables(self, noises: Tensor, **kwargs):
        """Generic function to allow diversity between models in definition/computation of causal variables
        given the noises (i.e. the output of 'encode')"""
        X = self.causal_block(noises, self.tau)
        return X

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

    def get_representation(self, inputs:Tensor, **kwargs):
        """ returns a representation vector for the given inputs"""
        causal = kwargs.get('causal', False)
        codes = self.encode_mu(inputs, **kwargs)
        if causal: return self.get_causal_variables(codes, **kwargs)
        return codes

    def estimate_range_dim(self, dim:int, device:str, size=100, **kwargs):
        """Estimates the range of the selected causal variable by sampling from the prior"""

        codes = self.sample_noise_from_prior(device=device, num_samples=size).detach()
        X = self.get_causal_variables(codes, **kwargs).detach().reshape(-1, self.latent_size, self.xunit_dim)
        X_d = X[:, dim, :]  # (num_samples,) -
        ranges = [(torch.min(X_d[:,i]).detach().cpu().numpy().item(),
                 torch.max(X_d[:,i]).detach().cpu().numpy().item()) for i in range(self.xunit_dim)]
        return ranges

