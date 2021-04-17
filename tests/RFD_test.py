import os
import unittest
from datasets import RFD
from torch.utils.data import DataLoader
import torch
import torchvision

class TestRDF(unittest.TestCase):
    def setUp(self) -> None:
        self.root = '../datasets/robot_finger_datasets/'
        self.transform = torchvision.transforms.ToTensor()

    def test_load_train(self):
        RDF_dataset = RFD(root = self.root,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,)) # 9 labels

    def test_load_test_heldout(self):
        RDF_dataset = RFD(root = self.root,
                          heldout_colors=True,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,))

    def test_load_test_real(self):
        RDF_dataset = RFD(root = self.root,
                          real=True,
                          transform=self.transform)
        new_item = next(RDF_dataset.iterator)
        self.assertEqual(new_item[0].shape, (3,128,128))
        self.assertEqual(new_item[1].shape, (9,))

    def test_print(self):
        RDF_dataset = RFD(root = self.root,
                          transform=self.transform)
        print(RDF_dataset)
        # passed if nothing fails

    def test_dataloader_making(self):
        RDF_dataset = RFD(root = self.root,
                          transform=self.transform)
        tot = RDF_dataset.size
        train_split = 0.8
        train_num = int(train_split*tot)
        val_num = tot-train_num
        loader = DataLoader(RDF_dataset,
                   batch_size=50,
                   drop_last=True)
        print(len(loader))
        train, val= torch.utils.data.random_split(loader,
                                                  lengths = [15000, 5000],
                                                  generator=torch.Generator().manual_seed(42))

        test_set = DataLoader(val,
                   batch_size=50,
                   shuffle=True,
                   drop_last=True)


if __name__ == '__main__':
    unittest.main()
