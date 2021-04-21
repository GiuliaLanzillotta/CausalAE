""" Implementation of loading functions for 3dshpaes dataset """
from typing import Any, Optional, Callable
from matplotlib import pyplot as plt
import numpy as np
import h5py
import torch
import urllib
import os
from torchvision.datasets import VisionDataset
from . import gen_bar_updater


class Shapes3d(VisionDataset):
    """ 3dShapes dataset: original paper http://proceedings.mlr.press/v80/kim18b.html
    Simulated dataset.
    Samples generated from 6 independently samples latent factors:
    floor hue: 10 values linearly spaced in [0, 1]
    wall hue: 10 values linearly spaced in [0, 1]
    object hue: 10 values linearly spaced in [0, 1]
    scale: 8 values linearly spaced in [0, 1]
    shape: 4 values in [0, 1, 2, 3]
    orientation: 15 values linearly spaced in [-30, 30]

    The data is stored in a HDF5 file with the following fields:
    images: (480000 x 64 x 64 x 3, uint8) RGB images.
    labels: (480000 x 6, float64) Values of the latent factors.
    """
    url = "https://console.cloud.google.com/storage/browser/_details/3d-shapes/3dshapes.h5"
    images_file = 'images.pt'
    labels_file = 'labels.pt'
    shape = (3, 64, 64)
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}

    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super(Shapes3d, self).__init__(root, transform=transform,
                                       target_transform=target_transform)
        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')
        self.data = torch.load(os.path.join(self.processed_folder, self.images_file))
        self.targets = torch.load(os.path.join(self.processed_folder, self.labels_file))



    def read_source_file(self):
        """ Reads the .h5 file into training and test arrays"""
        filename = self.url.rpartition('/')[2] #3dshapes.h5
        fpath = str.join("/", [self.raw_folder, filename])
        print("Reading " + filename)
        dataset = h5py.File(fpath, 'r')
        print(dataset.keys())
        images = dataset['images']  # array shape [480000,64,64,3], uint8 in range(256)
        labels = dataset['labels']  # array shape [480000,6], float64
        with open(os.path.join(self.processed_folder, self.images_file), 'wb') as f:
            torch.save(images, f)
        with open(os.path.join(self.processed_folder, self.labels_file), 'wb') as f:
            torch.save(labels, f)


    def download(self):
        """ Downloading source files (.h5)"""
        if self._check_exists():
            return
        os.makedirs(self.raw_folder, exist_ok=True)
        os.makedirs(self.processed_folder, exist_ok=True)
        filename = self.url.rpartition('/')[2] #3dshapes.h5
        fpath = str.join("/", [self.raw_folder, filename])
        # --------- download source file
        if not os.path.exists(fpath):
            try:
                print('Downloading ' + self.url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    self.url, fpath, reporthook=gen_bar_updater())
            except (urllib.error.URLError, IOError) as e:  # type: ignore[attr-defined]
                url = self.url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                urllib.request.urlretrieve(
                    url, fpath, reporthook=gen_bar_updater())
        # --------- load it and save it
        self.read_source_file()

    def __getitem__(self, index: int) -> Any:
        img = self.data[index,:]
        target = self.targets[index, :]
        img = img / 255. # normalise values to range [0,1]
        img = img.astype(np.float32)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]

    @property
    def raw_folder(self) -> str:
        # raw folder should be ("./datasets/3dShapes/Shapes3d/raw
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        # raw folder should be ("./datasets/3dShapes/Shapes3d/processed
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.images_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.labels_file)))


    # methods for sampling unconditionally/conditionally on a given factor
    # copied from deepmind dataset loading tutorial:
    # https://github.com/deepmind/3d-shapes/blob/master/3dshapes_loading_example.ipynb
    def get_index(self, factors):
        """ Converts factors to indices in range(num_data)
        Args:
          factors: np array shape [6,batch_size].
                   factors[i]=factors[i,:] takes integer values in
                   range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[i]]).

        Returns:
          indices: np array shape [batch_size].
        """
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self._FACTORS_IN_ORDER))):
            indices += factors[factor] * base
            base *= self._NUM_VALUES_PER_FACTOR[name]
        return indices

    def sample_batch(self, batch_size, fixed_factor, fixed_factor_value):
        """ Samples a batch of images with fixed_factor=fixed_factor_value, but with
            the other factors varying randomly.
        Args:
          batch_size: number of images to sample.
          fixed_factor: index of factor that is fixed in range(6).
          fixed_factor_value: integer value of factor that is fixed
            in range(_NUM_VALUES_PER_FACTOR[_FACTORS_IN_ORDER[fixed_factor]]).

        Returns:
          batch: images shape [batch_size,64,64,3]
        """
        factors = np.zeros([len(self._FACTORS_IN_ORDER), batch_size],
                           dtype=np.int32)
        for factor, name in enumerate(self._FACTORS_IN_ORDER):
            num_choices = self._NUM_VALUES_PER_FACTOR[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        factors[fixed_factor] = fixed_factor_value
        indices = self.get_index(factors)
        ims = []
        for ind in indices:
            im = self.data[ind]
            im = np.asarray(im)
            ims.append(im)
        ims = np.stack(ims, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3])


