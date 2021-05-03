import os
import unittest
from tests import RFDIterable
from torch.utils.data import DataLoader
import torch
import torchvision

class TestRDFIterable(unittest.TestCase):
    def setUp(self) -> None:
        self.root = '../datasets/robot_finger_datasets/'
        self.transform = torchvision.transforms.ToTensor()

    def test_load_train(self):
        RDF_dataset = RFDIterable(root = self.root,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,)) # 9 labels

    def test_load_test_heldout(self):
        RDF_dataset = RFDIterable(root = self.root,
                          heldout_colors=True,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,))

    def test_load_test_real(self):
        RDF_dataset = RFDIterable(root = self.root,
                          real=True,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,))

    def test_print(self):
        RDF_dataset = RFDIterable(root = self.root,
                          transform=self.transform)
        print(RDF_dataset)
        # passed if nothing fails

    def test_make_dataloader(self):
        RDF_dataset = RFDIterable(root = self.root,
                                  transform=self.transform)
        loader = DataLoader(RDF_dataset,
                            batch_size=500,
                            shuffle=False,
                            drop_last=True)
        item = loader.__iter__().__next__()
        print(item[0].shape, item[1].shape)


if __name__ == '__main__':
    unittest.main()
