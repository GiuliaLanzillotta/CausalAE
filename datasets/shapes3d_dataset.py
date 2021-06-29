""" Implementation of loading functions for 3dshpaes dataset """
from typing import Any, Optional, Callable
from .disentanglement_datasets import DisentanglementDataset
import numpy as np
import h5py
import torch
import urllib
import pickle
import os
from torchvision.datasets import VisionDataset
from .utils import gen_bar_updater, transform_discrete_labels


class Shapes3d(VisionDataset, DisentanglementDataset):
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


    url = "https://storage.googleapis.com/3d-shapes/3dshapes.h5"
    images_file = 'images.pt'
    labels_file = 'labels.pt'
    factors_dict_file = 'factors.pkl'
    shape = (3, 64, 64)
    _FACTORS_IN_ORDER = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                         'orientation']
    _NUM_VALUES_PER_FACTOR = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10,
                              'scale': 8, 'shape': 4, 'orientation': 15}
    key_pad_len = 5

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

        if not self._check_downloaded():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        # --------- load it
        self.images, self.labels = self.read_source_file()
        # --------- load factors dictionary
        self.factors = self.factorise()

        print("Dataset loaded.")


    def categorise_labels(self, labels:np.ndarray):
        """Turn labels into categorical variables, and store them as integers.
        labels: numpy array of shape (num_samples, num_factors) containing the labels."""
        print("Categorising labels...")
        ranges = [np.linspace(0,1,10),
                  np.linspace(0,1,10),
                  np.linspace(0,1,10),
                  np.linspace(0,1,8),
                  range(4),
                  np.linspace(-30,30,15)]
        categorised = transform_discrete_labels(labels, ranges)
        return categorised

    def __repr__(self):
        """ Str representation of the dataset """
        head = "Dataset {0} info".format(self.__class__.__name__)
        body = ["Size = {0}".format(len(self)), "Factors of variation : "]
        for n,v_num in self._NUM_VALUES_PER_FACTOR.items():
            line = n+" with "+str(v_num)+" values"
            body.append(line)
        last_line = "For more info see original paper: http://proceedings.mlr.press/v80/kim18b.html"
        lines = [head] + [" " * 2 + line for line in body] + [last_line]
        return '\n'.join(lines)


    def read_source_file(self):
        """ Reads the .h5 file into training and test arrays"""
        filename = self.url.rpartition('/')[2] #3dshapes.h5
        fpath = str.join("/", [self.raw_folder, filename])
        print("Reading " + filename)
        dataset = h5py.File(fpath, 'r')
        images = dataset['images'][:]  # array shape [480000,64,64,3], uint8 in range(256)
        labels = dataset['labels'][:]  # array shape [480000,6], float64
        labels = self.categorise_labels(labels)
        return images, labels

    def check_and_substitute(self, factors:np.ndarray, other_factors:np.ndarray, index:int):
        """Checks if all the factors in the factors array exists in the dataset
        - overrides the implementation given in DisentanglementDataset superclass"""
        return factors


    def factorise(self):
        """ Creates the factors dictionary, i.e. a dictionary storing the index relative
        to any factor combination. This is the core of sample_observations_from_factors."""
        os.makedirs(self.processed_folder, exist_ok=True)
        filename = self.factors_dict_file
        fpath = str.join("/", [self.processed_folder, filename])
        # --------- download source file
        if self._check_factorised():
            print("Factors dictionary already created. Proceed to reading.")
            with open(fpath, 'rb') as f:
                factors = pickle.load(f)
        else:
            print("Creating factors dictionary.")
            # 1. for each label
            # 2. extract all the numbers
            # 3. pad the numbers and convert to string
            # 4. use result as dictionary key
            factors = {self.convert_to_key(self.labels[i]): i for i in range(len(self))}
            with open(fpath, 'wb') as f:
                pickle.dump(factors, f, pickle.HIGHEST_PROTOCOL)

        return factors

    def download(self):
        """ Downloading source files (.h5)"""
        os.makedirs(self.raw_folder, exist_ok=True)
        # --------- download source file
        if self._check_downloaded():
            print("Files already downloaded. Proceed to reading.")
        else:
            filename = self.url.rpartition('/')[2] #3dshapes.h5
            fpath = str.join("/", [self.raw_folder, filename])
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
            print("Done!")

    def _check_downloaded(self):
        filename = self.url.rpartition('/')[2] #3dshapes.h5
        fpath = str.join("/", [self.raw_folder, filename])
        return os.path.exists(fpath)

    def _check_factorised(self):
        """Checking the existence of the factors dictionary."""
        filename = self.factors_dict_file
        fpath = str.join("/", [self.processed_folder, filename])
        return os.path.exists(fpath)

    def __getitem__(self, index: int) -> Any:
        img = np.asarray(self.images[index])
        target = torch.tensor(np.asarray(self.labels[index]))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.images.shape[0]

    @property
    def raw_folder(self) -> str:
        # raw folder should be ("./datasets/Shapes3d/Shapes3d/raw
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        # raw folder should be ("./datasets/Shapes3d/Shapes3d/processed
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    @property
    def num_factors(self):
        return len(self._FACTORS_IN_ORDER)

    @property
    def factors_names(self):
        return self._FACTORS_IN_ORDER

    @property
    def factors_num_values(self):
        return self._NUM_VALUES_PER_FACTOR


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
        lbls = []
        for ind in indices:
            im = self.images[ind]
            lbl = self.labels[ind]
            im = np.asarray(im)
            lbl = np.asarray(lbl)
            ims.append(im)
            lbls.append(lbl)
        ims = np.stack(ims, axis=0)
        lbls = np.stack(lbls, axis=0)
        ims = ims / 255. # normalise values to range [0,1]
        ims = ims.astype(np.float32)
        return ims.reshape([batch_size, 64, 64, 3]), lbls


