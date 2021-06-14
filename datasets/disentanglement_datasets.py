""" Generalisation of Disentanglement datasets.
A disentanglement dataset is a dataset generated from a two-step generative model
for which we have access to ground truth on the factors of variation."""
import random
from abc import ABCMeta, abstractmethod, ABC
from collections import Iterable

import numpy as np

class DisentanglementDataset(ABC):

    def __init__(self):
        self.factors = None
        self.key_pad_len = None

    @abstractmethod
    @property
    def num_factors(self):
        raise NotImplementedError


    @abstractmethod
    @property
    def factors_names(self):
        """List of strings with name of factors"""
        raise NotImplementedError

    @abstractmethod
    @property
    def factors_num_values(self):
        """Dictionary: factor name -> factor size"""
        raise NotImplementedError()


    def convert_to_key(self, factor:Iterable):
        """ Coverts factor (list of int) to key for Factors dictionary"""
        return "".join([format(number, '0'+str(self.key_pad_len)+'d') for number in factor])

    def revert_to_int(self, factor:str):
        """ Convert factor in key format back to integer."""
        numbers = [int(factor[i*self.key_pad_len:(i+1)*self.key_pad_len]) for i in range(self.num_factors)]
        return np.array(numbers) #TODO: check typying here

    def sample_observations_from_partial_factors(self, factors, fixed_indices, num_samples=1):
        """Sample a batch of observations X given a batch of partially fixed factors Y.
        factors: list/array of factors Y already converted in strings (as obtained from sample_factors)
        resample_indices: indices of dimensions to resample randomly.
        num_samples: number of factors to sample from each partial entry
        Note: the dataset could be non-complete, which means that the random resampling has
        to be done within the available combinations."""
        #TODO: fix here
        def check_partial_factor(word, factor):
            partial_word = list(map(word.__getitem__, [idx*self.key_pad_len for idx in fixed_indices))
            partial_factor = list(map(factor.__getitem__, fixed_indices*self.key_pad_len))
            return partial_word == factor

        available_factors = self.factors.keys()
        samples = [random.sample(
            list(filter(lambda key: check_partial_factor(key, factor), available_factors)),
            num_samples) for factor in factors]
        observations = [self[self.factors[k]] for k in samples]
        #TODO: now we have a list of tensors: check the desired data type (probably tensor)
        return observations

    def sample_observations_from_factors(self, factors):
        """Sample a batch of observations X given a batch of factors Y.
        Factors : batch of labels - has to be numpy array.
        Returns batch of corresponding images."""
        n = factors.shape[0]
        #For i in 1:n
        # 1. transform factor to string to obtain dict key
        # 2. get corresponding dict item = image index
        # 3. get corresponding item from dataset
        items = [self[self.factors[self.convert_to_key(factors[i])]] for i in range(n)]
        #TODO: now we have a list of tensors: check the desired data type (probably tensor)
        return items

    def sample_factors(self, num):
        """Sample a batch of the latent factors."""
        #TODO: fix here
        factors = np.zeros(shape=(num, self.num_factors), dtype=np.int64)
        available_factors = self.factors.keys()
        samples = random.sample(available_factors, num)
        return factors
