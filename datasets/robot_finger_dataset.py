""" Implementation of loading functions for robot_finger_dataset"""
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union, Iterator
from torch.utils.data import IterableDataset
from torch.utils.data.dataset import T_co
from torchvision.transforms import ToPILImage,ToTensor
from PIL import Image
import itertools
import webdataset as wds
import numpy as np
import tarfile
import torch
import os, io


class RFD(IterableDataset):
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

    shape = (128,128,3)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]
    train_percentage = 0.9

    class RFD_items_iterator(Iterator):
        """ This iterator class accomplishes the following:
        - zipping together labels and images iterators
        - applying torch transformations and in general all preprocessing"""
        def __init__(self,
                     images_iterator:Iterator,
                     labels_iterator:Iterator,
                     transform: Optional[Callable] = None,
                     target_transform: Optional[Callable] = None):
            super().__init__()
            self.images_iterator = images_iterator
            self.labels_iterator = labels_iterator
            self.transform = transform
            self.target_transform = target_transform

        def __next__(self) -> Tuple:
            next_image = next(self.images_iterator)
            next_label = next(self.labels_iterator)
            if self.transform is not None:
                next_image = self.transform(next_image)
            if self.target_transform is not None:
                next_label = self.target_transform(next_label)
            return (next_image, next_label)



    def __init__(self,
                 root: str,
                 heldout_colors: bool =False,
                 real: bool=False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        """ Can load one of 3 datasets available:
        - the standard dataset
        - the heldout_colors test set
        - the real images test set
        Note: only one of the test sets can be loaded at a time.
        (Ideally this could be changed)
        """
        super(RFD, self).__init__()
        self.root = root
        print("====== Opening RFD ======")
        self.heldout_test = heldout_colors
        self.real_test = real
        self.origin_folder = self.raw_subfolders[0]
        print("===== Reading files =====")
        images, labels = self.load_files()
        self.iterator = self.RFD_items_iterator(images, labels, transform, target_transform)

    def load_files(self):
        """ Initialises dataset by loading files from disk"""
        # switching folder
        if self.heldout_test: folder_index=1
        elif self.real_test: folder_index=2
        else: folder_index=0
        raw_folder = self.raw_subfolders[folder_index]
        self.origin_folder=raw_folder
        if folder_index == 2:
            # no need to read .tar file
            path = os.path.join(self.raw_folder, raw_folder, raw_folder+"_images.npz")
            images = iter(np.load(path, allow_pickle=True)["images"])
        else:
            # need to split in test and training
            path = os.path.join(self.raw_folder, raw_folder, raw_folder+"_images.tar")
            images = iter(wds.Dataset(path).decode("pil").map(lambda tup: tup["png"]))
        # loading labels
        path = os.path.join(self.raw_folder, raw_folder, raw_folder+"_labels.npz")
        labels = iter(np.load(path, allow_pickle=True)["labels"])
        return images, labels

    def __iter__(self) -> Iterator:
        return self.iterator

    def read_dataset_info(self):
        # loading info
        path = os.path.join(self.raw_folder, self.origin_folder, self.origin_folder+"_info.npz")
        info = dict(np.load(path, allow_pickle=True))
        self.size = info["dataset_size"].item()
        self.info = info
        factor_values_dict = info["factor_values"].item()
        self.factor_names = list(factor_values_dict.keys())
        self.factor_values_num = list(info["num_factor_values"])

    def __repr__(self):
        self.read_dataset_info()
        head = "Dataset "+ self.__class__.__name__ + self.origin_folder +" info"
        body = ["Size = " + str(self.size), "Factors of variation : "]
        for n,v_num in zip(self.factor_names, self.factor_values_num):
            line = n+" with "+str(v_num)+" values"
            body.append(line)
        lines = [head] + [" " * 2 + line for line in body]
        return '\n'.join(lines)

    @property
    def raw_folder(self) -> str:
        # this will be something like './datasets/robot_finger_datasets/RFD/raw'
        return os.path.join(self.root, self.__class__.__name__, 'raw')
