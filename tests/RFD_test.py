import unittest
from datasets import RFD

class TestRDF(unittest.TestCase):
    def setUp(self) -> None:
        self.root = '../datasets/robot_finger_datasets/'

    def test_load_train(self):
        RDF_dataset = RFD(root = self.root)
        self.assertEqual((next(RDF_dataset.iterator))[0].shape, (3,128,128))


if __name__ == '__main__':
    unittest.main()
