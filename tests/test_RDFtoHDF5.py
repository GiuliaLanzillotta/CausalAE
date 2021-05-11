import unittest
from . import RFDtoHDF5
import h5py
import os

class TestRFDtoHDF5(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        #"transforming" the data format
        cls.toHDF5 = RFDtoHDF5(chunksize=32)

    def test_write(self):
        num_chunks_to_write = 10
        filename = "RFD_test.h5"
        self.toHDF5._write_to_HDF5(filename, num_chunks_to_write)
        with h5py.File(self.toHDF5.filepath + filename, 'r') as f:
            shape_imgs = f['images'].shape
            shape_lbls = f['labels'].shape
        self.assertEqual(shape_imgs[0], num_chunks_to_write*32)
        self.assertEqual(shape_lbls[0], num_chunks_to_write*32)
        self.assertEqual(shape_imgs[1:], (3,128,128))

    def test_overWrite(self):
        filename = f"RFD{0}.h5"
        # write once
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
        self.toHDF5(overwrite=True, length=32*40, num_files=10) # 40/10 = 4 chunks per file
        for n in range(10):
            file_name = f"RFD{n}.h5"
            self.assertTrue(os.path.exists(self.toHDF5.filepath + file_name))
            with h5py.File(self.toHDF5.filepath + file_name, 'r') as f:
                images = f['images']
                self.assertTrue(images.shape[0], 32*4)


if __name__ == '__main__':
    unittest.main()
