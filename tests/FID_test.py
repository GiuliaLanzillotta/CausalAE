import unittest
import numpy as np
from . import FIDScorer, DatasetLoader
unittest.TestLoader.sortTestMethodsUsing = None

class FidTest(unittest.TestCase):
    def setUp(self) -> None:
        self.args = {"dataset_name": "celeba",
                "batch_size": 144, # Better to have a square number
                "test_batch_size": 144}
        self.batch_size = 144

    def test_model_creation(self):

        self.loader = DatasetLoader(self.args)
        """ Test correct model creation"""
        self._fidscorer = FIDScorer()
        #passes if not failing


    def test_everything(self):
        #todo: reformat code
        self.loader = DatasetLoader(self.args)
        """ Test correct model creation"""
        self._fidscorer = FIDScorer()
        #passes if not failing
        self._fidscorer.start_new_scoring(self.batch_size*2)
        self.assertEqual(self._fidscorer.generated.shape, (self.batch_size*2, 2048))
        input_imgs, _ = next(self.loader.train.__iter__())
        self._fidscorer.get_activations(input_imgs, input_imgs)
        self.assertEqual(self._fidscorer.start_idx, self.batch_size)
        input_imgs2, _ = next(self.loader.train.__iter__())
        self._fidscorer.get_activations(input_imgs, input_imgs)
        self.assertEqual(self._fidscorer.start_idx, self.batch_size*2)
        mu_generated, sigma_generated, mu_originals, sigma_originals = self._fidscorer.calculate_activation_statistics()
        self.assertEqual(mu_generated, mu_originals)
        self.assertEqual(sigma_generated, sigma_originals)
        fid = self._fidscorer.calculate_fid()
        self.assertEqual(fid, 0)


    def test_start_new_scoring(self):
        self._fidscorer.start_new_scoring(self.batch_size*2)
        self.assertEqual(self._fidscorer.generated.shape, (self.batch_size*2, 2048))

    def test_get_activations(self):
        input_imgs, _ = next(self.loader.train.__iter__())
        self._fidscorer.get_activations(input_imgs, input_imgs)
        self.assertEqual(self._fidscorer.start_idx, self.batch_size)
        input_imgs2, _ = next(self.loader.train.__iter__())
        self._fidscorer.get_activations(input_imgs, input_imgs)
        self.assertEqual(self._fidscorer.start_idx, self.batch_size*2)

    def test_calculate_activation_statistics(self):
        mu_generated, sigma_generated, mu_originals, sigma_originals = self._fidscorer.calculate_activation_statistics()
        self.assertEqual(mu_generated, mu_originals)
        self.assertEqual(sigma_generated, sigma_originals)

    def test_calculate_fid(self):
        fid = self._fidscorer.calculate_fid()
        self.assertEqual(fid, 0)


if __name__ == '__main__':
    unittest.main()
