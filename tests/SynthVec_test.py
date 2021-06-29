import unittest

import numpy as np

from . import SynthVec

class SynthVecTestCase(unittest.TestCase):

    def test_discrete_generation(self):
        dataset = SynthVec("./datasets/SynthVec/", 10, 100,
                           allow_continuous=False, allow_discrete=True,
                           generate=True, overwrite=True,
                           test=False, noise=True, verbose=True)
        self.assertTrue(dataset.labels.shape[1]==10)
        self.assertTrue(dataset.data.shape[1]==100)
        for factor in dataset.factors_names:
            self.assertTrue(dataset.factors_num_values[factor]<=15)

    def test_continuous_generation(self):
        dataset = SynthVec("./datasets/SynthVec/", 10, 100,
                           allow_continuous=True, allow_discrete=False,
                           generate=True, overwrite=True,
                           test=False, noise=True, verbose=True)
        self.assertTrue(dataset.labels.shape[1]==10)
        self.assertTrue(dataset.data.shape[1]==100)
        for factor in dataset.factors_names:
            self.assertTrue(dataset.factors_num_values[factor]==20)


    def test_continuous_and_discrete_generation(self):
        dataset = SynthVec("./datasets/SynthVec/", 10, 100,
                           allow_continuous=True, allow_discrete=True,
                           generate=True, overwrite=True,
                           test=False, noise=True,
                           verbose=True)
        self.assertTrue(dataset.labels.shape[1]==10)
        self.assertTrue(dataset.data.shape[1]==100)
        self.assertTrue(sum(dataset.metadata["discrete"])<=10)
        tot_values = 0
        for factor in dataset.factors_names:
            tot_values += dataset.factors_num_values[factor]
        self.assertTrue(tot_values<200)

    def test_labelling(self):
        # load the dataset
        dataset = SynthVec("./datasets/SynthVec/", 10, 100,
                           allow_continuous=True, allow_discrete=True,
                           generate=False, overwrite=True,
                           test=False, noise=True,
                           verbose=True)
        X1,Y1,N1,X2,Y2,N2,metadata = dataset.read_source_files()
        self.assertTrue(np.array_equal(N1, dataset.original_labels))
        dataset = SynthVec("./datasets/SynthVec/", 10, 100,
                           allow_continuous=True, allow_discrete=True,
                           generate=False, test=False, noise=False,
                           verbose=True)
        self.assertTrue(np.array_equal(Y1, dataset.original_labels))



if __name__ == '__main__':
    unittest.main()
