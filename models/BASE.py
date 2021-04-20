""" Implements interface for generative autoencoders """
from abc import ABCMeta, abstractmethod, ABC
from torch import Tensor


class GenerativeAE(ABC):

    @abstractmethod
    def encode(self, inputs:Tensor):
        raise NotImplementedError

    @abstractmethod
    def decode(self, noise:Tensor):
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_prior(self):
        raise NotImplementedError

    @abstractmethod
    def sample_noise_from_posterior(self, inputs:Tensor):
        raise NotImplementedError

    @abstractmethod
    def forward(self, inputs:Tensor):
        raise NotImplementedError


