import time
import unittest
from . import RFDh5, RFD
from torchvision.transforms import ToTensor
import numpy as np

class RFDh5Test(unittest.TestCase):
    """Note: to run this test please first create a set of .h5 files with RFDtoHDF5
    by running the RFDtoHDF5 tests """
    root = './datasets/robot_finger_datasets/'

    def test_relative_index(self):
        dataset = RFDh5(self.root)
        # in 'test_train_test_split' we create 8 train files, each with 4 chunks of size 32
        # hence every 32*4 = 128 items a new partition should be considered
        indices = [32, 32*5, 32*9, 32*13, 32*17, 32*21, 32*25, 32*29]
        expected = [0, 1, 2, 3, 4, 5, 6, 7]
        actual = [dataset.get_relative_index(i) for i in indices]
        self.assertTrue(all([actual[j]==(expected[j],32) for j in range(8)]))


    def test_init_partitions(self):
        dataset = RFDh5(self.root)
        self.assertTrue(all([dataset.partitions_dict[i][1] == 128 for i in range(8)]))

    def test_train(self):
        dataset = RFDh5(self.root)
        img, lbl = dataset[20]
        self.assertEqual(img.shape, (3, 128, 128))
        self.assertEqual(lbl.shape, (9,))


    def test_test(self):
        dataset = RFDh5(self.root, test=True)
        self.assertEqual(len(dataset.partitions_dict.keys()), 2)
        img, lbl = dataset[20]
        self.assertEqual(img.shape, (3, 128, 128))
        self.assertEqual(lbl.shape, (9,))

    def test_heldout(self):
        dataset = RFDh5(self.root, heldout_colors=True)
        self.assertEqual(len(dataset.partitions_dict.keys()), 1)
        img, lbl = dataset[20]
        self.assertEqual(img.shape, (3, 128, 128))
        self.assertEqual(lbl.shape, (9,))

    def test_real(self):
        dataset = RFDh5(self.root, real=True)
        img, lbl = dataset[20]
        self.assertEqual(img.shape, (3, 128, 128)) #TODO: fix here
        self.assertEqual(lbl.shape, (9,))

    @unittest.skip("The access times are close enough but not almost equals")
    def test_random_access(self):
        dataset = RFDh5(self.root, heldout_colors=False)
        start = time.time()
        _ = dataset[4]
        end = time.time()
        delta1 = end-start

        start = time.time()
        _ = dataset[1000]
        end = time.time()
        delta2 = end-start

        start = time.time()
        _ = dataset[500]
        end = time.time()
        delta3 = end-start

        self.assertAlmostEqual(delta1, delta2)
        self.assertAlmostEqual(delta2, delta3)

    @unittest.skip("test can only be run if the whole .tar archive is expanded")
    def test_indices_order(self):
        """ Here we want to test that the indices order is kept the same in the .h5 files"""
        train = RFDh5(self.root)
        test = RFDh5(self.root, test=True)
        rfd = RFD(root=self.root, transform=ToTensor)
        train_indices = [0, 33, 130, 900]
        test_indices = [0, 33, 200]
        test_indices_base = 32*32 #=1024
        for i in train_indices:
            img_dataset = train.__getitem__(i)
            # loading from file
            img_archive = rfd.__getitem__(i)
            self.assertEqual(img_archive, img_dataset)
        for i in test_indices:
            img_dataset = test.__getitem__(i)
            # loading from file
            img_archive = rfd.__getitem__(i + test_indices_base)
            self.assertEqual(img_archive, img_dataset)

    def test_get_labels(self):
        train = RFDh5(self.root)
        _, lbl = train[20]
        labels = train.get_labels()
        self.assertTrue(np.array_equal(lbl.cpu().numpy(), labels[20]))
        test = RFDh5(self.root, test=True)
        _, lbl = test[20]
        labels = test.get_labels()
        self.assertTrue(np.array_equal(lbl.cpu().numpy(), labels[20]))


if __name__ == '__main__':
    unittest.main()
