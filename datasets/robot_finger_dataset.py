""" Implementation of loading functions for robot_finger_dataset"""
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage,ToTensor
from PIL import Image
import numpy as np
import tarfile
import torch
import os, io

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

    shape = (128,128,3)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]
    train_percentage = 0.9

    training_file = 'training.pt'
    standard_test_file = 'test.pt'
    heldout_test_file = 'heldout_test.pt'
    real_test_file = 'real_test.pt'



    def __init__(self, root: str, train: bool = True,
                 heldout_colors: bool =False,
                 real: bool=False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 load: bool = False,
                 only_subset:bool = True) -> None:
        """ Can load one of 4 datasets available:
        - the training dataset
        - the standard test dataset
        - the heldout_colors test set
        - the real images test set
        Note: only one of the test sets can be loaded at a time.
        (Ideally this could be changed)
        """
        super(RFD, self).__init__(root, transform=transform,
                                  target_transform=target_transform)

        self.train = train  # training set or test set
        self.heldout_test = heldout_colors
        self.real_test = real
        self.origin_folder = self.raw_subfolders[0]
        self.only_subset = only_subset

        if load: self.load_from_disk()
        #TODO: print statements
        if not all(self._check_exists()):
            raise RuntimeError('Dataset not found.' +
                               ' You can use load=True to load it from file')

        if self.train:
            data_file = self.training_file
        elif self.heldout_test:
            data_file = self.heldout_test_file
            self.origin_folder = self.raw_subfolders[1]
        elif self.real_test_file:
            data_file = self.real_test_file
            self.origin_folder = self.raw_subfolders[2]
        else:
            data_file = self.standard_test_file

        self.data, self.targets = torch.load(os.path.join(self.processed_folder, data_file))


    @staticmethod
    def read_tar(path, limit:int):
        """Reads tar file, navigating all subdirectories and storing all images
        in a list of PIlImages"""
        print("Opening "+str(path))
        tar = tarfile.open(path)
        to_tensor = ToTensor()
        #navigating subdirectories
        imgs = []
        files_read = 0
        for member in tar:
            if member.isreg(): #regular file
                files_read +=1
                image = tar.extractfile(member)
                image = image.read()
                image = to_tensor(Image.open(io.BytesIO(image)))
                imgs.append(image)
                if files_read%100000==0: print(str(files_read)+ " files read.")
                if limit and files_read==limit: break
        return imgs

    def read_dataset_info(self):
        # loading info
        path = os.path.join(self.raw_folder, self.origin_folder, self.origin_folder+"_info.npz")
        info = dict(np.load(path, allow_pickle=True))
        self.info = info
        factor_values_dict = self.info["factor_values"].item()
        self.factor_names = list(factor_values_dict.keys())
        self.factor_values_num = list(self.info["num_factor_values"])

    def __repr__(self):
        standard_descrpt = super(RFD).__repr__()
        self.read_dataset_info()
        head = "Dataset "+ self.__class__.__name__ + self.origin_folder +" info"
        body = ["Factors of variation : "]
        for n,v_num in zip(self.factor_names, self.factor_values_num):
            line = n+" with "+str(self.factor_values_num)+" values"
            body.append(line)
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join([standard_descrpt, '\n'.join(lines)])

    def get_train_test_split(self, images, labels):
        """ Performs train-test splitting of standard dataset """
        size = len(images)
        train_samples = int(self.train_percentage*size)
        # randomly sample 'train_samples' indices
        #TODO: check whether shuffling makes sense
        idx = np.random.permutation(size)
        train_indices = idx[:train_samples]
        test_indices = idx[train_samples:]
        train_images = [images[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        test_images = [images[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        return train_images,train_labels,test_images,test_labels

    def load_from_disk(self):
        """ One-time use function: loads the files from disk opening the various .tar, .npz
        files and stores the content as torch tensors. """

        LIMIT = 100000
        os.makedirs(self.processed_folder, exist_ok=True)
        exist_flags = self._check_exists()
        if all(exist_flags): return
        print("Preprocessing...")
        for i,folder in enumerate(self.raw_subfolders):
            if exist_flags[i]: continue
            # loading images
            if i==2: # true images stored in .npz
                path = os.path.join(self.raw_folder, folder, folder+"_images.npz")
                images = np.load(path, allow_pickle=True)["images"]
            else:
                # open .tar files
                path = os.path.join(self.raw_folder, folder, folder+"_images.tar")
                images = self.read_tar(path, limit=LIMIT)
            # loading labels
            path = os.path.join(self.raw_folder, folder, folder+"_labels.npz")
            labels = np.load(path, allow_pickle=True)["labels"]
            if self.only_subset: labels=labels[:LIMIT]#num_data_points x 9
            if i==0: # for the standard dataset we split in train and test sets
                # train-test split
                train_images,train_labels,test_images,test_labels = self.get_train_test_split(images, labels)
                training_set = (train_images, train_labels)
                test_set = (test_images, test_labels)
                # saving to file
                torch.save(training_set, os.path.join(self.processed_folder, self.training_file))
                torch.save(test_set, os.path.join(self.processed_folder, self.standard_test_file))
                del training_set, test_set
            else:
                dataset = (images, labels)
                torch.save(dataset, os.path.join(self.processed_folder,
                                                 self.heldout_test_file if i==1
                                                 else self.real_test_file))
                del dataset

        print("Done!")

    def __getitem__(self, index: int) -> Any:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

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

    def _check_exists(self) -> List:
        flags = [os.path.exists(os.path.join(self.processed_folder, self.training_file)) and \
                   os.path.exists(os.path.join(self.processed_folder, self.standard_test_file)),
                 os.path.exists(os.path.join(self.processed_folder, self.heldout_test_file)),
                 os.path.exists(os.path.join(self.processed_folder, self.real_test_file))]

        return flags