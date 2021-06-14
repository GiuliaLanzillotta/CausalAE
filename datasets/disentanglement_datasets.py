""" Generalisation of Disentanglement datasets.
A disentanglement dataset is a dataset generated from a two-step generative model
for which we have access to ground truth on the factors of variation."""
from abc import ABCMeta, abstractmethod, ABC

class DisentanglementDataset(ABC):

    @abstractmethod
    @property
    def num_factors(self):
        raise NotImplementedError

    @abstractmethod
    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @abstractmethod
    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y."""
        raise NotImplementedError()
