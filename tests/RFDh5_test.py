import time
import unittest
from . import RFDh5

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

    def test_random_access(self):
        dataset = RFDh5(self.root, heldout_colors=False)
        start = time.time()
        _ = dataset[4]
        end = time.time()
        delta1 = end-start

        start = time.time()
        _ = dataset[1000] #TODO: fix here
        end = time.time()
        delta2 = end-start

        start = time.time()
        _ = dataset[500]
        end = time.time()
        delta3 = end-start

        self.assertAlmostEqual(delta1, delta2)
        self.assertAlmostEqual(delta2, delta3)




if __name__ == '__main__':
    unittest.main()
