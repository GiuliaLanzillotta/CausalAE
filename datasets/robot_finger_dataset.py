""" Implementation of loading functions for robot_finger_dataset"""
from typing import Any, Callable, Optional, Tuple, Iterator

import numpy
import numpy as np
import torch
import torchvision.datasets
import webdataset as wds
from PIL import Image
import random
import h5py
import os
from .utils import gen_bar_updater
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice

class RFD(torchvision.datasets.VisionDataset):
    """  The Robot Finger Dataset - Vision dataset version """
    shape = (3, 128,128)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]

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
        (Ideally this could be changed) """
        super(RFD, self).__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        self.heldout_test = heldout_colors
        self.real_test = real
        self.origin_folder = self.raw_subfolders[0]
        if heldout_colors: self.origin_folder = self.raw_subfolders[1]
        print("====== Opening RFD Dataset ======")
        if real:
            self.origin_folder = self.raw_subfolders[2]
            self.images = self.read_real_images()
        self.labels, self.info = self.read_labels_and_info()
        self.size = self.info["dataset_size"].item()


    def __repr__(self):
        """ Str representation of the dataset """
        factor_values_dict = self.info["factor_values"].item()
        factor_names = list(factor_values_dict.keys())
        factor_values_num = list(self.info["num_factor_values"])
        head = "Dataset {0}_{1} info".format(self.__class__.__name__, self.origin_folder)
        body = ["Size = {0}".format(self.size), "Factors of variation : "]
        for n,v_num in zip(factor_names, factor_values_num):
            line = n+" with "+str(v_num)+" values"
            body.append(line)
        last_line = "For more info see original paper: https://arxiv.org/abs/2010.14407"
        lines = [head] + [" " * 2 + line for line in body] + [last_line]
        return '\n'.join(lines)

    def read_real_images(self):
        """ Loading function only for real test dataset images"""
        path = str.join("/", [self.raw_folder, self.origin_folder, self.origin_folder])
        images = np.load(path+"_images.npz", allow_pickle=True)["images"]
        return images

    def read_labels_and_info(self):
        """ Opens the labels and info .npz files and stores them in the class"""
        # loading labels
        path = str.join("/", [self.raw_folder, self.origin_folder, self.origin_folder])
        labels = np.load(path+"_labels.npz", allow_pickle=True)["labels"]
        info = dict(np.load(path+"_info.npz", allow_pickle=True))
        return labels, info

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

    def __getitem__(self, index: int) -> Any:
        target = self.labels[index]
        if self.real_test: img = self.images[index]
        else: img = self.read_image_from_archive(index)

        if self.transform is not None:
            img = self.transform(img)

        # this could fail if image is not a torch tensor
        img = img.view(self.shape)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return self.size

    @property
    def raw_folder(self) -> str:
        # this will be something like './datasets/robot_finger_datasets/RFD/raw'
        return self.root +self.__class__.__name__+"/"+'raw'


class RFDIterable(IterableDataset):
    """ The Robot Finger Dataset - Iterable version
    This version directly works with the .tar files obtained by the first
    un-tarring of the dataset (finger_images.tar, finger_heldout_colors.tar, finger_real.tar)
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

    shape = (3, 128,128)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]
    train_percentage = 0.9

    class RFD_standard_iterator(Iterator):
        """ This iterator class accomplishes the following:
        - zipping together labels and images iterators
        - applying torch transformations and in general all preprocessing"""
        def __init__(self,
                     images_iterator:Iterator,
                     labels:numpy.ndarray,
                     images_path:str,
                     transform: Optional[Callable] = None,
                     target_transform: Optional[Callable] = None,
                     name:str="RFD_std"):
            super().__init__()
            self.path = images_path
            self.images_iterator = images_iterator
            self.labels = labels
            self.transform = transform
            self.target_transform = target_transform
            self.name = name

        def __next__(self) -> Tuple:
            try: next_item = next(self.images_iterator)
            except Exception as e:
                # iterator is finished, starting again
                print(self.name)
                print("Reached the end of iterator: rolling it back to beginning.")
                self.images_iterator = RFDIterable.initialise_web_dataset(self.path)
                next_item = next(self.images_iterator)
            next_idx, next_image = next_item
            next_label = torch.tensor(self.labels[int(next_idx)], requires_grad=False)
            if self.transform is not None:
                next_image = self.transform(next_image)
            if self.target_transform is not None:
                next_label = self.target_transform(next_label)

            return next_image, next_label

    class RFD_real_set_iterator(Iterator):
        """ This iterator class accomplishes the following:
        - zipping together labels and images iterators
        - applying torch transformations and in general all preprocessing"""
        def __init__(self,
                     images:numpy.ndarray,
                     labels:numpy.ndarray,
                     transform: Optional[Callable] = None,
                     target_transform: Optional[Callable] = None,
                     name:str="RFD_real"):
            super().__init__()
            self.images = images
            self.labels = labels
            self.images_iterator = iter(images)
            self.labels_iterator = iter(labels)
            self.transform = transform
            self.target_transform = target_transform
            self.name = name

        def __next__(self) -> Tuple:
            try: next_image = next(self.images_iterator)
            except Exception as e:
                # iterator is finished, starting again
                print(self.name)
                print("Reached the end of iterator: rolling it back to beginning.")
                self.images_iterator = iter(self.images)
                self.labels_iterator = iter(self.labels)
                next_image = next(self.images_iterator)
            next_label = torch.tensor(next(self.labels_iterator), requires_grad=False)
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
        """
        super(RFDIterable, self).__init__()
        self.root = root
        print("====== Opening RFD ======")
        self.heldout_test = heldout_colors
        self.real_test = real
        self.origin_folder = self.raw_subfolders[0]
        print("===== Reading files =====")
        images, labels = self.load_files() # this call will create the class variable 'tar_path'

        if real: self.iterator = self.RFD_real_set_iterator(images, labels, transform,
                                                            target_transform, name="TEST RFD")
        else: self.iterator = self.RFD_standard_iterator(images, labels, self.tar_path,
                                                         transform, target_transform,
                                                         name="TRAIN RFD" if not self.heldout_test else "VALID RFD")

        print(self)

    @staticmethod
    def initialise_web_dataset(path):
        itr = iter((wds.Dataset(path).decode("pil")
                    .map(lambda tup: (tup["__key__"].split("/")[-1], tup["png"]))  # e.g. (00012, PIL_IMAGE-png format)
                    .shuffle(10000))) # shuffling blocks of 1k images and pre-batching
        return itr


    def load_files(self):
        """ Initialises dataset by loading files from disk and (in case of .tar saving format) opening the web Dataset"""
        # switching folder
        if self.heldout_test: folder_index=1
        elif self.real_test: folder_index=2
        else: folder_index=0
        raw_folder = self.raw_subfolders[folder_index]
        self.origin_folder=raw_folder # save the root folder for the dataset (needed later)
        if folder_index == 2:
            # no need to read .tar file
            path =self.raw_folder+"/"+raw_folder+"/"+raw_folder+"_images.npz"
            images = np.load(path, allow_pickle=True)["images"]
        else:
            self.tar_path = self.raw_folder+"/"+raw_folder+"/"+raw_folder+"_images.tar"
            images = self.initialise_web_dataset(self.tar_path)
        # loading labels
        path = self.raw_folder+"/"+raw_folder+"/"+raw_folder+"_labels.npz"
        labels = np.load(path, allow_pickle=True)["labels"]
        return images, labels

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            return self.iterator
        # in a worker process
        worker_id = worker_info.id
        n_workers = worker_info.num_workers
        return islice(self.iterator, worker_id, None, n_workers)

    def __len__(self):
        return self.size

    def read_dataset_info(self):
        # loading info
        path = self.raw_folder +"/"+self.origin_folder+"/"+self.origin_folder+"_info.npz"
        info = dict(np.load(path, allow_pickle=True))
        self.size = info["dataset_size"].item()
        self.info = info
        factor_values_dict = info["factor_values"].item()
        self.factor_names = list(factor_values_dict.keys())
        self.factor_values_num = list(info["num_factor_values"])

    def __repr__(self):
        self.read_dataset_info()
        head = "Dataset "+ self.__class__.__name__ + "_"+self.origin_folder +" info"
        body = ["Size = " + str(self.size), "Factors of variation : "]
        for n,v_num in zip(self.factor_names, self.factor_values_num):
            line = n+" with "+str(v_num)+" values"
            body.append(line)
        lines = [head] + [" " * 2 + line for line in body]
        return '\n'.join(lines)

    @property
    def raw_folder(self) -> str:
        # this will be something like './datasets/robot_finger_datasets/RFD/raw'
        return self.root +self.__class__.__name__[:3]+"/"+'raw'



class RFDtoHDF5(object):
    """ Switches the dataset format from .tar to hdf5
    Exploits the RFD dataset class to do so.
    The class will write the dataset to the ./processed/RFD.h5 file under the specified root."""

    #TODO: standard + heldout + test
    #TODO: look into splitting

    read_root = './datasets/robot_finger_datasets/'
    save_root = './datasets/robot_finger_datasets/'
    filepath = save_root + "RFD/processed/"


    def __init__(self, chunksize:int=32):
        self.chunksize = chunksize
        self.init_RFD_dataloader()
        os.makedirs(self.filepath, exist_ok=True)

    def init_RFD_dataloader(self):
        transform = transforms.ToTensor()
        standard_set = RFD(self.read_root, transform=transform) #in torch tensor format
        self.dataset = DataLoader(standard_set, batch_size=self.chunksize, shuffle=True)

    @staticmethod
    def compute_files_partitions(length, num_files):
        len_per_file = (length//num_files)
        lens_per_files = [len_per_file]*num_files
        if len_per_file*num_files<length: #last file bigger
            lens_per_files[-1] += (length - len_per_file*num_files)
        return lens_per_files

    def compute_num_chunks_per_file(self, file_len):
        num_chunks = file_len//self.chunksize
        if len(self.dataset.dataset)%self.chunksize!=0:
            num_chunks+=1
        return num_chunks

    def _write_to_HDF5(self, filename:str, num_chunks:int):
        """ Test version of the write function
        num_chunks: number of chunks to read/write"""
        chunks_count = 1
        row_count = self.chunksize
        notify_every = 20
        updater = gen_bar_updater()
        # collect the first batch
        imgs, lbls = next(iter(self.dataset))
        with h5py.File(self.filepath+filename, 'w') as f:
            #create the datasets
            images = f.create_dataset('images', shape=imgs.shape, maxshape=(None,)+imgs.shape[1:],
                                      chunks=imgs.shape, dtype=np.float)
            images[:] = imgs.cpu().detach().numpy()
            labels = f.create_dataset('labels', shape=lbls.shape, maxshape=(None,)+lbls.shape[1:],
                                      chunks=lbls.shape, dtype=np.float)
            labels[:] = lbls.cpu().detach().numpy()
            # iterate over the rest
            for imgs, lbls in self.dataset:
                # Resize the dataset to accommodate the next chunk of rows
                images.resize(row_count + imgs.shape[0], axis=0)
                labels.resize(row_count + lbls.shape[0], axis=0)
                # Write the next chunk
                images[row_count:] = imgs.cpu().detach().numpy()
                labels[row_count:] = lbls.cpu().detach().numpy()
                # Increment the row count
                row_count += imgs.shape[0] #less than chunksize if last batch
                chunks_count+=1
                if chunks_count%notify_every==0:
                    updater(chunks_count, self.chunksize, min(num_chunks*self.chunksize, len(self.dataset.dataset)))
                if chunks_count==num_chunks: return

    def __call__(self, overwrite=False, **kwargs):
        num_files = kwargs.get('num_files',10)
        self.num_files = num_files
        partitions = self.compute_files_partitions(kwargs.get('length', len(self.dataset.dataset)), num_files)
        print("Writing RFD dataset to .h5 file")
        for n in range(num_files):
            file_name = f"RFD{n}.h5"
            if os.path.exists(self.filepath+file_name):
                if not overwrite:
                    print(f"file {file_name} already there and overwrite=False. Nothing to do.")
                    break
                else:
                    print("Removing existing file.")
                    os.remove(self.filepath+file_name)
            print(f'Writing file number {n}')
            self._write_to_HDF5(file_name, self.compute_num_chunks_per_file(partitions[n]))



class RFDh5(torchvision.datasets.VisionDataset):
    """ Implementing RA dataset for RFD based on .h5 storage"""
    shape = (3, 128,128)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]
    raw_files = [10, 1, 1] #TODO: see here
    #TODO: extend to heldout and real test dataset

    def __init__(self,
                 root: str,
                 heldout_colors: bool =False,
                 real: bool=False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root)
        self.heldout = heldout_colors
        self.real = real
        print("====== Opening RFD Dataset ======")
        self.info = self.read_info()
        self.partitions = self.shuffle_indices_up_to(self.num_files)
        self.partition = -1 #no partition loaded

    @staticmethod
    def shuffle_indices_up_to(num):
        indices = list(range(num))
        random.Random(11).shuffle(indices)
        return indices

    def get_file_from_index(self, index):
        avg_len_per_file = (len(self)//self.num_files)
        partition_idx = self.partitions[index//avg_len_per_file]
        return partition_idx

    def open_partition(self, number):
        if self.partition == number: return
        filename = f"RFD_{self.set_name}{number}.h5" #TODO: make sure to save this way
        with h5py.File(self.processed_folder + filename, 'r') as f:
            images = f['images']
            labels = f['labels']
        self.partition = number
        self.images = images
        self.labels = labels
        self.partition_indices = self.shuffle_indices_up_to(images.shape[0])

    def __getitem__(self, index: int) -> Any:
        par_idx = self.get_file_from_index(index)
        self.open_partition(par_idx)
        avg_len_per_file = (len(self)//self.num_files)
        relative_index = index%avg_len_per_file
        if index >= avg_len_per_file*self.num_files:
            relative_index = index - avg_len_per_file*self.num_files
        index = self.partition_indices[relative_index]
        image = self.images[index]
        labels = self.labels[index]


    def __len__(self) -> int:
        return self.size #TODO: see here too

    def read_info(self):
        """ Opens the labels and info .npz files and stores them in the class"""
        # loading labels
        path = str.join("/", [self.raw_folder, self.set_name, self.set_name])
        info = dict(np.load(path+"_info.npz", allow_pickle=True))
        self.size = info["dataset_size"].item()
        #TODO: extract interesting properties from the info here
        return info

    @property
    def raw_folder(self) -> str:
        return self.root + 'RFD/raw'

    @property
    def num_files(self) -> int:
        if self.heldout: return self.raw_files[1]
        if self.real: return self.raw_files[2]
        return self.raw_files[0]

    @property
    def set_name(self) -> str:
        if self.heldout: return self.raw_subfolders[1]
        if self.real: return self.raw_subfolders[2]
        return self.raw_subfolders[0]

    @property
    def processed_folder(self) -> str:
        return self.root + 'RFD/processed'




