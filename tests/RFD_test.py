import os
import unittest
from tests import RFD
from torch.utils.data import DataLoader
import torch
import torchvision

class TestRFD(unittest.TestCase):
    def setUp(self) -> None:
        # Note: test to be run on the cluster
        self.root = '/cluster/scratch/glanzillo/robot_finger_datasets/'
        self.transform = torchvision.transforms.ToTensor()

    def test_load_standard(self):
        standard_set = RFD(self.root, transform=self.transform)
        img234 = standard_set.__getitem__(234)
        self.assertEqual(img234[0].shape, (3,128,128))
        self.assertEqual(standard_set.labels.shape[0], standard_set.size)


    def test_load_test_heldout(self):
        heldout_set = RFD(self.root,
                          heldout_colors=True,
                          transform=self.transform)
        img234 = heldout_set.__getitem__(234)
        self.assertEqual(img234[0].shape, (3,128,128))
        self.assertEqual(heldout_set.labels.shape[0], heldout_set.size)

    def test_load_test_real(self):
        real_set = RFD(self.root,
                       real=True,transform=self.transform)
        img234 = real_set.__getitem__(234)
        self.assertEqual(img234[0].shape, (3,128,128))
        self.assertEqual(real_set.labels.shape[0], real_set.size)

    def test_print(self):
        RDF_dataset = RFD(self.root, transform=self.transform)
        print(RDF_dataset)
        # passed if nothing fails

    def test_read_image_from_archive(self):
        standard_set = RFD(self.root, transform=self.transform)
        max_len = len(str(standard_set.size-1))
        for i in range(max_len):
            idx = 3*(10**i)
            img = standard_set.__getitem__(idx)
            # passed if nothing fails

if __name__ == '__main__':
    unittest.main()
