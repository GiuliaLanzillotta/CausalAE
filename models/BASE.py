""" Implements interface for generative autoencoders """
from abc import ABCMeta, abstractmethod, ABC
from torch import Tensor


class GenerativeAE(ABC):

    latent_size = None

    @abstractmethod
    def encode(self, inputs:Tensor) -> Tensor:
        """ returns all the encoder's output (noise and code included) for given input"""
        raise NotImplementedError

    @abstractmethod
    def encode_mu(self, inputs:Tensor) -> Tensor:
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


