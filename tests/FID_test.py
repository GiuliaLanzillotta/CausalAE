import unittest
import numpy as np
from . import FIDScorer, DatasetLoader
unittest.TestLoader.sortTestMethodsUsing = None

class FidTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.args = {"dataset_name": "celeba",
                "batch_size": 144, # Better to have a square number
                "test_batch_size": 144}
        cls.batch_size = 144

    def test_everything(self):
        """ Statefule testing with subTests """
        loader = DatasetLoader(self.args)
        fidscorer = FIDScorer()
        #passes if not failing
        with self.subTest():
            fidscorer.start_new_scoring(self.batch_size*2)
            self.assertEqual(fidscorer.generated.shape, (self.batch_size*2, 2048))

        with self.subTest(msg="Activations update for first batch"):
            input_imgs, _ = next(loader.train.__iter__())
            fidscorer.get_activations(input_imgs, input_imgs)
            self.assertEqual(fidscorer.start_idx, self.batch_size)

        with self.subTest(msg="Activations update for second batch"):
            input_imgs2, _ = next(loader.train.__iter__())
            fidscorer.get_activations(input_imgs, input_imgs)
            self.assertEqual(fidscorer.start_idx, self.batch_size*2)

        with self.subTest("Activation statistics not equal"):
            mu_generated, sigma_generated, mu_originals, sigma_originals = fidscorer.calculate_activation_statistics()
            self.assertTrue(all(mu_generated==mu_originals))
            self.assertTrue(all((sigma_generated==sigma_originals).reshape(-1)))

        with self.subTest("FID value not 0"):
            fid = fidscorer.calculate_fid()
            eps = 10e-5
            self.assertTrue(fid<eps)

if __name__ == '__main__':
    unittest.main()
