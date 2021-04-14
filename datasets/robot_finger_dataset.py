""" Implementation of loading functions for robot_finger_dataset"""
from typing import Any, Callable, Dict, IO, List, Optional, Tuple, Union
from torchvision.datasets import VisionDataset
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

        self.origin_folder = self.raw_subfolders[0]
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


    def read_tar(self, path):
        """Reads tar file, navigating all subdirectories and storing all images
        in a torch tensor"""
        print("Opening "+str(path))
        tar = tarfile.open(path)
        #navigating subdirectories
        imgs = []
        files_read = 0
        for member in tar.getmembers():
            if member.isreg(): #regular file
                files_read +=1
                image = tar.extractfile(member.name)
                image = image.read()
                image = Image.open(io.BytesIO(image))
                imgs.append(torch.Tensor(image.getdata(), requires_grad=False).view(self.shape))
                if files_read%100000==0: print(str(files_read)+ " files read.")
        tensor = torch.stack(imgs)
        return tensor

    def read_dataset_info(self):
        # loading info
        path = os.path.join(self.raw_folder, self.origin_folder, self.origin_folder+"_info.npz")
        info = dict(np.load(path, allow_pickle=True))
        self.info = info
        factor_values_dict = self.info["factor_values"].item()
        self.factor_names = list(factor_values_dict.keys())
        self.factor_values_num = list(self.info["num_factor_values"])

    def __repr__(self):
        standard_descrpt = super.__repr__(self)
        self.read_dataset_info()
        head = "Dataset "+ self.__class__.__name__ + " info"
        body = ["Factors of variation : "]
        for n,v_num in zip(self.factor_names, self.factor_values_num):
            line = n+" with "+str(self.factor_values_num)+" values"
            body.append(line)
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join([standard_descrpt, lines])

    def get_train_test_split(self, tensor, labels):
        """ Performs train-test splitting of standard dataset """
        size = tensor.shape[0]
        train_samples = int(self.train_percentage*size)
        # randomly sample 'train_samples' indices
        idx = torch.randperm(size)
        train_indices = idx[:train_samples]
        test_indices = idx[train_samples:]
        train_tensor = tensor[train_indices]
        train_labels = labels[train_indices]
        test_tensor = tensor[test_indices]
        test_labels = labels[test_indices]
        return train_tensor,train_labels,test_tensor,test_labels

    def load_from_disk(self):
        """ One-time use function: loads the files from disk opening the various .tar, .npz
        files and stores the content as torch tensors. """
        os.makedirs(self.processed_folder, exist_ok=True)
        print("Preprocessing...")
        # open .tar files
        for i,folder in enumerate(self.raw_subfolders):
            # loading images
            if i==2: # true images stored in .npz
                path = os.path.join(self.raw_folder, folder, folder+"_images.npz")
                tensor = torch.tensor(np.load(path, allow_pickle=True)["images"], requires_grad=False)
            else:
                path = os.path.join(self.raw_folder, folder, folder+"_images.tar")
                tensor = self.read_tar(path)
            # loading labels
            path = os.path.join(self.raw_folder, self.origin_folder, self.origin_folder+"_labels.npz")
            labels = torch.tensor(np.load(path, allow_pickle=True)["labels"], requires_grad=False) #num_data_points x 9
            if i==0: # for the standard dataset we split in train and test sets
                # train-test split
                train_tensor,train_labels,test_tensor,test_labels = self.get_train_test_split(tensor, labels)
                training_set = (train_tensor, train_labels)
                test_set = (test_tensor, test_labels)
                # saving to file
                with open(os.path.join(self.processed_folder, self.training_file), 'wb') as f:
                    torch.save(training_set, f)
                with open(os.path.join(self.processed_folder, self.standard_test_file), 'wb') as f:
                    torch.save(test_set, f)
            else:
                set = (tensor, labels)
                with open(os.path.join(self.processed_folder, self.heldout_test_file if i==1 else self.real_test_file), 'wb') as f:
                    torch.save(set, f)

        print("Done!")


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