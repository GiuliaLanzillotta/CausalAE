import unittest
from . import RFDtoHDF5
import h5py
import os

class TestRFDtoHDF5(unittest.TestCase):

    def setUp(self) -> None:
        #"transforming" the data format
        self.toHDF5 = RFDtoHDF5(chunksize=32, heldout_colors=False)

    def test_write(self):
        self.toHDF5.split = False
        self.toHDF5.init_RFD_dataloader()
        num_chunks_to_write = 10
        filename = "RFD_testing.h5"
        self.toHDF5._write_to_HDF5(filename, num_chunks_to_write)
        with h5py.File(self.toHDF5.filepath + filename, 'r') as f:
            shape_imgs = f['images'].shape
            shape_lbls = f['labels'].shape
        self.assertEqual(shape_imgs[0], num_chunks_to_write*32)
        self.assertEqual(shape_lbls[0], num_chunks_to_write*32)
        self.assertEqual(shape_imgs[1:], (3,128,128))

    def test_overWrite(self):
        filename = f"RFD__{0}.h5"
        # write once
        self.toHDF5.split = False
        self.toHDF5.init_RFD_dataloader()
        self.toHDF5._write_to_HDF5(filename, num_chunks=10)
        # overwrite
        self.toHDF5(overwrite=True, length=32*5, num_files=1)
        # load and look
        with h5py.File(self.toHDF5.filepath + filename, 'r') as f:
            shape_imgs = f['images'].shape
            shape_lbls = f['labels'].shape
        self.assertEqual(shape_imgs[0], 5*32)
        self.assertEqual(shape_lbls[0], 5*32)
        self.assertEqual(shape_imgs[1:], (3,128,128))

    def test_write_multiple_files(self):
        self.toHDF5.split = False
        self.toHDF5.init_RFD_dataloader()
        self.toHDF5(overwrite=True, length=32*40, num_files=10, id='testing') # 40/10 = 4 chunks per file
        for n in range(10):
            file_name = f"RFD{n}.h5"
            self.assertTrue(os.path.exists(self.toHDF5.filepath + file_name))
            with h5py.File(self.toHDF5.filepath + file_name, 'r') as f:
                images = f['images']
                self.assertTrue(images.shape[0], 32*4)

    def test_compute_partitions(self):
        lens_per_files = self.toHDF5.compute_files_partitions(32*40,10)
        expected = [4]*10
        self.assertTrue(all([lens_per_files[i]==expected[i] for i in range(10)]))
        lens_per_files = self.toHDF5.compute_files_partitions(32*40 + 30,10)
        expected[-1] += 1
        self.assertTrue(all([lens_per_files[i]==expected[i] for i in range(10)]))
        lens_per_files = self.toHDF5.compute_files_partitions(32*40,7)
        expected = [5]*7
        expected[-1] = 10
        self.assertTrue(all([lens_per_files[i]==expected[i] for i in range(7)]))


    def test_train_test_split(self):
        self.toHDF5(overwrite=True, length=32*40, num_files=10) # 40/10 = 4 chunks per file
        # 32x40 in 8/2 split ---> 32 chunks train, 8 chunks test and 8 files train, 2 files test
        for n in range(8): # training
            file_name = f"RFD_train_{n}.h5"
            self.assertTrue(os.path.exists(self.toHDF5.filepath + file_name))
            with h5py.File(self.toHDF5.filepath + file_name, 'r') as f:
                images = f['images']
                self.assertTrue(images.shape[0], 32*4)
        for n in range(2):
            file_name = f"RFD_test_{n}.h5"
            self.assertTrue(os.path.exists(self.toHDF5.filepath + file_name))
            with h5py.File(self.toHDF5.filepath + file_name, 'r') as f:
                images = f['images']
                self.assertTrue(images.shape[0], 32*4)

    def test_heldout_set(self):
        toHDF5 = RFDtoHDF5(chunksize=32, heldout_colors=True)
        toHDF5(overwrite=True, length=32*4+20, num_files=1)
        file_name = f"RFD_HC_{0}.h5"
        with h5py.File(self.toHDF5.filepath + file_name, 'r') as f:
            images = f['images']
            self.assertTrue(images.shape[0], 32*4 + 20)



if __name__ == '__main__':
    unittest.main()
