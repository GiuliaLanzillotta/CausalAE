import unittest

from torch.utils.data.dataset import T_co

from . import DisentanglementDataset
from torch.utils.data import Dataset
import torch
import numpy as np

class RandomDataset(DisentanglementDataset, Dataset):

    shape = (10,10,1)

    def __init__(self):
        super().__init__()
        self.key_pad_len = 3
        self.labels, self.X = self.create_data()
        self.factors = self.create_factors()

    def __len__(self):
        return 1000

    @staticmethod
    def make_observation(label):
        """ Creates the observation given the label"""
        vector = np.expand_dims(label,1) # shape (10,1)
        x = np.dot(vector, vector.T) # shape (10,10)
        return x

    def create_data(self):
        labels = np.random.randint(5, size=(len(self), self.num_factors))
        labels[0] = np.zeros(self.num_factors, dtype=np.int)
        labels[1] = np.asarray([0]*5+[1]*5)
        labels[2] = np.ones(self.num_factors, dtype=np.int)
        X = np.zeros((len(self),)+self.shape, dtype=np.int)
        for i,lbl in enumerate(labels):
            X[i] = np.expand_dims(self.make_observation(lbl),2)
        return labels,torch.Tensor(X)

    def create_factors(self):
        factors = {self.convert_to_key(self.labels[i]): i for i in range(len(self))}
        return factors

    @property
    def num_factors(self):
        return 10

    @property
    def factors_names(self):
        return ["factor"+str(i) for i in range(10)]

    @property
    def factors_num_values(self):
        return [5] * 10


    def __getitem__(self, index) -> T_co:
        return self.X[index]


class DisentanglementDatasetsTest(unittest.TestCase):

    dataset = RandomDataset()

    def test_sample_factors(self):
        _factors = self.dataset.sample_factors(2, numeric_format=False)
        for f in _factors:
            obs = self.dataset[self.dataset.factors[f]]
            self.assertTrue(np.array_equal(obs.squeeze(2).numpy(),self.dataset.make_observation(self.dataset.revert_to_int(f))))

    def test_sample_pairs(self):
        idx, obs1, obs2 = self.dataset.sample_pairs_observations(2)
        self.assertTrue(obs1[0,idx,idx,0]==obs2[0,idx,idx,0])
        self.assertTrue(obs1[1,idx,idx,0]==obs2[1,idx,idx,0])


    def test_sample_obs_from_factors(self):
        match_base = self.dataset.convert_to_key(np.zeros(10, dtype=np.int))
        obs = self.dataset.sample_observations_from_factors([match_base])
        self.assertTrue(np.array_equal(obs[0],torch.zeros(10,10,1, dtype=torch.int)))

    def sample_obs_from_partial_factors(self):
        match_base = self.dataset.convert_to_key(np.zeros(10))
        indices = [0,1] # basically matching all the factors that start with 2 zeros
        lbl, obs = self.dataset.sample_observations_from_partial_factors([match_base],
                                                                         [indices], num_samples=1)
        self.assertTrue(obs[0][0,0,0]==0)
        self.assertTrue(obs[0][0,1,0]==0)
        self.assertTrue(obs[0][1,0,0]==0)
        self.assertTrue(obs[0][1,1,0]==0)


    def test_all_similar_to(self):
        match_base = self.dataset.convert_to_key(np.zeros(10, dtype=np.int))
        indices = [0,1] # basically matching all the factors that start with 2 zeros
        res = self.dataset._all_similar_to(match_base, indices)
        self.assertTrue(self.dataset.convert_to_key(np.zeros(10, dtype=np.int)) in res)
        self.assertTrue(self.dataset.convert_to_key(np.asarray([0]*5+[1]*5)) in res)
        self.assertFalse(self.dataset.convert_to_key(np.ones(10, dtype=np.int)) in res)




if __name__ == '__main__':
    unittest.main()
