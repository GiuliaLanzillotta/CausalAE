""" Implementation of loading functions for robot_finger_dataset"""
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from PIL import Image
import tarfile
import torch
import os

class RFD(VisionDataset):
    """ The Robot Finger Dataset
    ----------------------------
    3 different datasets:
    - finger: simulated images
    - finger_heldout_colors: test set with held-out cube colors
    - finger_real: test set with real images (to be used for sim2real evaluation)

    ----------------------------
    The class offers the following methods and properties:
    -
    -
    """

    training_file = 'training.pt'
    standard_test_file = 'test.pt'
    heldout_test_file = 'heldout_test.pt'
    real_test_file = 'real_test.pt'



    def __init__(self, root: str, train: bool = True,
                 heldout_colors: bool =False,
                 real: bool=False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 load: bool = False) -> None:
        """ Can load one of 4 datasets available:
        - the training dataset
        - the standard test dataset
        - the heldout_colors test set
        - the real images test set
        """
        super().__init__(root, transform, target_transform)

        self.train = train  # training set or test set
        self.heldout_test = heldout_colors
        self.real_test = real
        if load: self.load_from_disk()
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use load=True to load it from file')

        if self.train:
            data_file = self.training_file
        elif self.heldout_test:
            data_file = self.heldout_test_file
        elif self.real_test_file:
            data_file = self.real_test_file
        else:
            data_file = self.standard_test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))


    def load_from_disk(self):
        """ One-time use function: loads the files from disk opening the various .tar, .npz
        files and stores the content as torch tensors. """
        os.makedirs(self.processed_folder, exist_ok=True)



    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        # this will be something like './datasets/robot_finger_datasets/RFD/raw'
        return os.path.join(self.root, self.__class__.__name__, 'raw')

    @property
    def processed_folder(self) -> str:
        # this will be something like './datasets/robot_finger_datasets/RFD/processed'
        return os.path.join(self.root, self.__class__.__name__, 'processed')

    def _check_exists(self) -> bool:
        return (os.path.exists(os.path.join(self.processed_folder,
                                            self.training_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.standard_test_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.heldout_test_file)) and
                os.path.exists(os.path.join(self.processed_folder,
                                            self.real_test_file)))