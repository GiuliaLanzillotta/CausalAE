""" Generalisation of Disentanglement datasets.
A disentanglement dataset is a dataset generated from a two-step generative model
for which we have access to ground truth on the factors of variation."""
import logging.config
import random
from abc import ABCMeta, abstractmethod, ABC
from collections import Iterable
from typing import List, Union
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

    @abstractmethod
    def categorise_labels(self, labels):
        """Turn labels into categorical variables, and store them as integers"""
        pass

    def convert_to_key(self, factor:Iterable):
        """ Coverts factor (list of int) to key for Factors dictionary"""
        return "".join([format(number, '0'+str(self.key_pad_len)+'d') for number in factor])


    def revert_to_int(self, factor:str):
        """ Convert factor in key format back to integer."""
        numbers = [int(factor[i*self.key_pad_len:(i+1)*self.key_pad_len]) for i in range(self.num_factors)]
        return np.array(numbers) #TODO: check typying here

    def check_partial_factor(self, first, second, fixed_indices:List[int]) -> bool:
        """Checks equivalence between first and second factor (both in str key format)
        only on the fixed_indices"""
        first_num = self.revert_to_int(first)
        second_num = self.revert_to_int(second)
        return all([first_num[i]==second_num[i] for i in fixed_indices])


    def _one_similar_to(self, factor, other_factor:str, fixed_indices:List[int]):
        available_factors = self.factors.keys()
        entry = next(f for f in available_factors
                     if (self.check_partial_factor(f, factor, fixed_indices) and f!=other_factor))
        return entry

    def _all_similar_to(self, factor, fixed_indices:List[int]):
        """Collects all the entries similar to the given one on the fixed_indices"""
        available_factors = self.factors.keys()
        similar_entries = list(filter(lambda key: self.check_partial_factor(key, factor, fixed_indices), available_factors))
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
            #TODO: getting all of those similar is a waste
            similar_entries = self._all_similar_to(factor,indices)
            if len(similar_entries)<num_samples: missed+=1
            else:
                labels.append(random.sample(similar_entries, num_samples))
                observations.append([self[self.factors[key]][0] for key in labels[-1]])
        logging.info("Total number of missed duplicates = "+str(missed))
        return labels, observations

    def sample_observations_from_factors(self, factors:Union[List[str], np.ndarray], numeric_format=False):
        """Sample a batch of observations X given a batch of factors Y.
        Factors : batch of labels - has to be numpy array.
        Returns batch of corresponding images."""
        n = len(factors)
        #For i in 1:n
        # 1. access i-th factor (key format)
        # 2. get corresponding dict item = image index
        # 3. get corresponding item from dataset
        items = [self[self.factors[factors[i]]][0] for i in range(n)] if not numeric_format \
            else [self[self.factors[self.convert_to_key(factors[i])]][0] for i in range(n)]
        return torch.stack(items)

    def sample_factors(self, num, numeric_format=False):
        """Sample a batch of the latent factors.
        Returns a list of factors in key string format by default. Converts them back to int numpy array otherwise."""
        available_factors = self.factors.keys()
        samples = random.sample(available_factors, num)
        if numeric_format: samples = np.vstack([self.revert_to_int(s) for s in samples]) #shape = (num, num_factors)
        return samples

    def check_and_substitute(self, factors:np.ndarray, other_factors:np.ndarray, index:int):
        """Checks if all the factors in the factors array exists in the dataset"""
        #TODO: override in complete datasets and simply return the input
        count=0
        for i in range(factors.shape[0]):
            factor = self.convert_to_key(factors[i])
            other_factor = self.convert_to_key(other_factors[i])
            try: self.factors[factor]
            except KeyError:
                #TODO: consider the case the given factor cannot be sampled
                # this only happens if for some factor value there exists only one example in the dataset
                new_factor = self._one_similar_to(factor, other_factor, [index]*factors.shape[1])
                factors[i] = self.revert_to_int(new_factor)
                count+=1
        print("Total of "+str(count)+" substitutions done.")
        return factors

    def sample_pairs_observations(self, num):
        """ Samples a batch of pairs of observations as used in BetaVAE disentanglement metric.
        -> only one factor index fixed for every pair"""
        first_factors = self.sample_factors(num, numeric_format=True)
        second_factors = self.sample_factors(num, numeric_format=True)
        index = np.random.randint(0,self.num_factors)
        second_factors[:,index] = first_factors[:,index]
        second_factors = self.check_and_substitute(second_factors, first_factors, index)
        obs1 = self.sample_observations_from_factors(first_factors, numeric_format=True)
        obs2 = self.sample_observations_from_factors(second_factors, numeric_format=True)
        return index, obs1, obs2
