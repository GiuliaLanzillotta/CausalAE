""" Python script to convert RFD POSIX storage into HDF5
Note: if you want to use the Random Access RFD dataset you first need to download the .tar archive
and run this script."""
from PIL import Image
import random
import h5py
import numpy as np


class To_HDF5(object):

    shape = (3, 128,128)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]

    def __init__(self, root_path):
        self.root_path = root_path
        self.size = ...
        self.shuffle_indices()

    def __call__(self, *args, **kwargs):
        pass

    def read_image_from_archive(self, index:int):
        """ Given the organised folder structure it is possible to read an image given
        its index. Returns the image as a PILImage."""
        # example path: .../images/7/3/9/0/9/739097.png
        max_len = len(str(self.size-1))
        idx = str(index)
        idx = '0'*(max_len-len(idx))+idx
        subpath = str.join("/",list(idx)[:-1])
        path = str.join("/", [self.raw_folder, self.origin_folder,"images",subpath,idx+".png"])
        img = Image.open(path)
        return img

    def shuffle_indices(self):
        """Shuffles the dataset creating a shuffled list of indices which will then be
        used to read the files from memory"""
        indices = list(range(self.size))
        random.Random(11).shuffle(indices)
        self.indices = indices


    def chunk_generator(self, chunksize:int=32):
        """Yields all chunks of images for the specified dataset
        #TODO: extend to divide between train, valid and test"""


