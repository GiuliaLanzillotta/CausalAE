""" Implements interface for generative autoencoders """
from abc import ABCMeta, abstractmethod, ABC
from torch import Tensor
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, VecSCMDecoder, VecSCM
from torch.nn import functional as F
from abc import ABCMeta, abstractmethod, ABC


class GenerativeAE(ABC):

    latent_size = None

    @abstractmethod
    def encode(self, inputs:Tensor) -> Tensor:
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
    def sample_noise_from_prior(self, num_samples:int) -> Tensor:
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


class HybridAE(GenerativeAE, nn.Module, ABC):
    """Generalisation of all the generative autoencoder models that adopt hybrid layers
    as stochastic layers"""
    @abstractmethod
    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self.latent_size = params["latent_size"]
        self.unit_dim = params["unit_dim"]
        self.N = params["latent_vecs"] # number of latent vectors to store for hybrid sampling
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_size, self.unit_dim, self.N)
        self.act = nn.Sigmoid()

    def encode(self, inputs: Tensor):
        codes = self.encoder(inputs)
        return codes

    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        """ returns latent code (not noise) for given input"""
        codes =  self.encode(inputs)
        _update_prior = kwargs.get("update_prior", False)
        _integrate = kwargs.get("integrate", False)
        if _update_prior: self.hybrid_layer.update_prior(codes, integrate=_integrate)
        return codes

    def sample_noise_from_prior(self, num_samples:int):
        """Equivalent to total hybridisation: every point is hybridised at the maximum level"""
        return self.hybrid_layer.sample_from_prior((num_samples,))

    def sample_noise_from_posterior(self, inputs: Tensor):
        codes = self.encode(inputs)
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate, update_prior=True, integrate=True)

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False, integrate=True) -> Tensor:
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes, integrate=integrate)
        output = self.decode(codes, activate)
        return  output

    def pixel_losses(self, X, X_hat):
        """ Computes both MSE and BCE loss for X and X_hat"""
        # mean over batch of the sum over all other dimensions
        MSE = torch.sum(F.mse_loss(self.act(X_hat), X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        BCE = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        return MSE,BCE

    def loss_function(self, *args):
        X_hat = args[0]
        X = args[1]
        MSE,BCE = self.pixel_losses(X,X_hat)
        return BCE, MSE

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        return self.hybrid_layer.prior_range


