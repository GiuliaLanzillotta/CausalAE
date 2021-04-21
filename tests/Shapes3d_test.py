import unittest
import torchvision
import unittest

import torchvision

from tests import Shapes3d


class TestShapes3d(unittest.TestCase):
    def setUp(self) -> None:
        # Note: test to be run on the cluster
        self.root = '../datasets/Shapes3d'
        self.transform = torchvision.transforms.ToTensor()

    def test_download_dataset(self):
        dataset = Shapes3d(self.root,
                           download=True,
                           transform=self.transform)
        img234 = dataset.__getitem__(234)
        self.assertEqual(img234[0].shape, (3,64,64))
        self.assertEqual(dataset.labels.shape[0], len(dataset))

    def test_load_dataset(self):
        dataset = Shapes3d(self.root,
                           download=False,
                           transform=self.transform)
        img234 = dataset.__getitem__(234)
        self.assertEqual(img234[0].shape, (3,64,64))
        self.assertEqual(dataset.labels.shape[0], len(dataset))

    def test_load_fixed_factor(self):
        dataset = Shapes3d(self.root,
                           download=True,
                           transform=self.transform)
        batch_size = 25
        fixed_factor_str = 'floor_hue' #@param ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
        fixed_factor = dataset._FACTORS_IN_ORDER.index(fixed_factor_str)  # floor hue
        fixed_factor_value = 0  # first value of floor hue - red
        img_batch, lbl_batch = dataset.sample_batch(batch_size, fixed_factor, fixed_factor_value)
        self.assertTrue(all([lbl_batch[i, fixed_factor]==fixed_factor_value for i in range(batch_size)]))


    def test_print(self):
        dataset = Shapes3d(self.root,
                           download=True,
                           transform=self.transform)
        print(dataset)
        # passed if nothing fails

if __name__ == '__main__':
    unittest.main()
