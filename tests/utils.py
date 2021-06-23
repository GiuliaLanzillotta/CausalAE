""" Set of utilities to use for testing"""
import random

from . import DisentanglementDataset
import numpy as np
import torch
from torch.utils.data import Dataset


class IdentityObservationsData(DisentanglementDataset, Dataset):
    """Data set where dummy factors """

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        y =  self.sample_factors(1)[0]
        x =  self.sample_observations_from_factors(y)
        return x,y

    @property
    def factors_names(self):
        return ["factor"+str(i) for i in range(10)]

    @property
    def num_factors(self):
        return 10

    @property
    def observation_shape(self):
        return 10

    @property
    def factors_num_values(self):
        return [10] * 10

    def sample_factors(self, num, numeric_format=False):
        """Sample a batch of factors Y."""
        return np.random.random_integers(10, size=(num, self.num_factors))

    def sample_observations_from_factors(self, factors:np.ndarray):
        """Sample a batch of observations X given a batch of factors Y."""
        return torch.Tensor(factors)

    def sample_pairs_observations(self, num):
        first_factors = torch.Tensor(self.sample_factors(num, numeric_format=False))
        second_factors = torch.Tensor(self.sample_factors(num, numeric_format=False))
        index = random.randint(0,self.num_factors-1)
        second_factors[:,index]=first_factors[:,index]
        return index, first_factors, second_factors


class DummyData(DisentanglementDataset, Dataset):
    """Dummy image data set of random noise used for testing."""

    def __len__(self):
        return 100000

    def __getitem__(self, item):
        y =  self.sample_factors(1)[0]
        x =  self.sample_observations_from_factors(y)[0]
        return x,y

    @property
    def factors_names(self):
        return ["factor"+str(i) for i in range(10)]

    @property
    def num_factors(self):
        return 10

    @property
    def factors_num_values(self):
        # 10 factors, each with 5 categories
        return [5] * 10

    @property
    def observation_shape(self):
        return [64, 64, 1]

    def sample_factors(self, num, numeric_format=False):
        """Sample a batch of factors Y."""
        return np.random.randint(5, size=(num, self.num_factors))

    def sample_observations_from_factors(self, factors:np.ndarray):
        """Sample a batch of observations X given a batch of factors Y."""
        return torch.Tensor(np.random.random_sample(size=(factors.shape[0], 64, 64, 1)))

    def sample_pairs_observations(self, num):
        index = random.randint(0,self.num_factors)
        obs1 = torch.Tensor(np.random.random_sample(size=(num, 64, 64, 1)))
        obs2 = torch.Tensor(np.random.random_sample(size=(num, 64, 64, 1)))
        return index,obs1,obs2
