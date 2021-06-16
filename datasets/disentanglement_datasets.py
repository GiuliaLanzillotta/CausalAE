""" Generalisation of Disentanglement datasets.
A disentanglement dataset is a dataset generated from a two-step generative model
for which we have access to ground truth on the factors of variation."""
import logging.config
import random
from abc import ABCMeta, abstractmethod, ABC
from collections import Iterable
from typing import List
import torch

import numpy as np

class DisentanglementDataset(ABC):

    def __init__(self):
        self.factors = None
        self.key_pad_len = None

    @property
    def num_factors(self):
        raise NotImplementedError

    @property
    def factors_names(self):
        """List of strings with name of factors"""
        raise NotImplementedError

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

    def _all_similar_to(self, factor, fixed_indices:List[int]):
        """Collects all the entries similar to the given one on the fixed_indices"""
        def check_partial_factor(word, factor) -> bool:
            """Checks equivalence between word and factor (both in str key format)
            only on the fixed_indices"""
            word_num = self.revert_to_int(word)
            factor_num = self.revert_to_int(factor)
            return all([word_num[i]==factor_num[i] for i in fixed_indices])
        available_factors = self.factors.keys()
        similar_entries = list(filter(lambda key: check_partial_factor(key, factor), available_factors))
        return similar_entries

    def sample_observations_from_partial_factors(self, factors, fixed_indices:List[List[int]], num_samples=1):
        """Sample a batch of observations X given a batch of partially fixed factors Y.
        factors: list of factors Y already converted in strings (as obtained from sample_factors)
        resample_indices: indices of dimensions to resample randomly.
        num_samples: number of factors to sample from each partial entry
        Note: the dataset could be non-complete, which means that the random resampling has
        to be done within the available combinations."""
        observations = []
        labels = []
        missed = 0
        for (factor,indices) in zip(factors,fixed_indices):
            similar_entries = self._all_similar_to(factor,indices)
            if len(similar_entries)<num_samples: missed+=1
            else:
                labels.append([random.sample(similar_entries, num_samples)])
                observations.append([self.factors[key] for key in labels[-1]])
        logging.info("Total number of missed duplicates = "+str(missed))
        return torch.stack(observations)

    def sample_observations_from_factors(self, factors:List[str]):
        """Sample a batch of observations X given a batch of factors Y.
        Factors : batch of labels - has to be numpy array.
        Returns batch of corresponding images."""
        n = len(factors)
        #For i in 1:n
        # 1. access i-th factor (key format)
        # 2. get corresponding dict item = image index
        # 3. get corresponding item from dataset
        items = [self[self.factors[factors[i]]] for i in range(n)]
        return torch.stack(items)

    def sample_factors(self, num, numeric_format=False):
        """Sample a batch of the latent factors.
        Returns a list of factors in key string format by default. Converts them back to int numpy array otherwise."""
        available_factors = self.factors.keys()
        samples = random.sample(available_factors, num)
        if numeric_format: samples = np.vstack([self.revert_to_int(s) for s in samples]) #shape = (num, num_factors)
        return samples

    def sample_pairs_observations(self, num):
        """ Samples a batch of pairs of observations as used in BetaVAE disentanglement metric.
        -> only one factor index fixed for every pair"""
        first_factors = self.sample_factors(num, numeric_format=False)
        index = random.randint(0,self.num_factors)
        fixed_indices = [index]*num # = [[index],[index], .... , [index]]
        _, observations = self.sample_observations_from_partial_factors(first_factors, fixed_indices, num_samples=2)
        observations1 = []
        observations2 = []
        for pair in observations:
            observations1.append(pair[0])
            observations2.append(pair[1])
        return index, torch.stack(observations1), torch.stack(observations2)

