import unittest
from . import HybridLayer
import torch

class TestHybridLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.hl = HybridLayer(10,1,100)
        self.vecs = torch.randn((200,10))
        self.hl.initialise_prior(self.vecs)

    def test_not_all_equal(self):
        """ Here we want to test that the sampled vectors are not all equal"""
        samples = self.hl.sample_from_prior((100,10))
        self.assertFalse(torch.equal(samples[0], samples[1]))

    def test_not_equal_prior(self):
        """ Here we want to make sure that the samples are a shuffled version of the prior
        (with very low probability a vector is left untouched)"""
        samples = self.hl.sample_from_prior((100,10))
        self.assertFalse(any([torch.equal(self.hl.prior[i], samples[0]) for i in range(100)]))


if __name__ == '__main__':
    unittest.main()
