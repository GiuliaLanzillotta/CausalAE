""" Implements interface for generative autoencoders """
from abc import ABCMeta, abstractmethod, ABC
from torch import Tensor


class GenerativeAE(ABC):

    @abstractmethod
    def encode(self, inputs:Tensor) -> Tensor:
        """ returns latent code (not noise) for given input"""
        raise NotImplementedError

    @abstractmethod
    def decode(self, noise:Tensor) -> Tensor:
        """ returns generated sample given noise"""
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_prior(self, num_samples:int) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_posterior(self, inputs:Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def generate(self, inputs:Tensor) -> Tensor:
        """ generates output from input"""
        raise NotImplementedError



