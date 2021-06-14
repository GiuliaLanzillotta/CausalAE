""" Implementation of loading functions for robot_finger_dataset"""
from math import ceil
from typing import Any, Callable, Optional, Tuple, Iterator

import numpy
import numpy as np
import torch
import torchvision.datasets
import webdataset as wds
from PIL import Image
import h5py
import os
from .utils import gen_bar_updater
from torch.utils.data import IterableDataset, DataLoader
from itertools import islice
from . import DisentanglementDataset

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
                     shuffle:bool=True,
                     name:str="RFD_std"):
            super().__init__()
            self.path = images_path
            self.images_iterator = images_iterator
            self.labels = labels
            self.transform = transform
            self.target_transform = target_transform
            self.shuffle = shuffle
            self.name = name

        def __next__(self) -> Tuple:
            try: next_item = next(self.images_iterator)
            except Exception as e:
                # iterator is finished, starting again
                print(self.name)
                print("Reached the end of iterator: rolling it back to beginning.")
                self.images_iterator = RFDIterable.initialise_web_dataset(self.path, self.shuffle)
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
                 target_transform: Optional[Callable] = None,
                 shuffle: bool = True) -> None:
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
        self.shuffle = shuffle
        print("===== Reading files =====")
        images, labels = self.load_files() # this call will create the class variable 'tar_path'

        if real: self.iterator = self.RFD_real_set_iterator(images, labels, transform,
                                                            target_transform, name="TEST RFD")
        else: self.iterator = self.RFD_standard_iterator(images, labels, self.tar_path,
                                                         transform, target_transform,
                                                         shuffle=shuffle,
                                                         name="TRAIN RFD" if not self.heldout_test else "VALID RFD")
        print(self)

    @staticmethod
    def initialise_web_dataset(path, shuffle):
        dataset = wds.Dataset(path).decode("pil").map(lambda tup: (tup["__key__"].split("/")[-1], tup["png"])) # e.g. (00012, PIL_IMAGE-png format)
        if shuffle: dataset = dataset.shuffle(10000) # shuffling blocks of 1k images and pre-batching
        itr = iter(dataset)
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
            images = self.initialise_web_dataset(self.tar_path, self.shuffle)
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

    train_test_split = 0.8 # note that also the number of files will be split with this percentage

    def __init__(self,
                 read_root:str= './datasets/robot_finger_datasets/',
                 save_root:str= './datasets/robot_finger_datasets/',
                 chunksize:int=32, heldout_colors: bool=False):
        self.read_root = read_root
        self.save_root = save_root
        self.filepath = save_root + "RFD/processed/"
        self.chunksize = chunksize
        self.heldout = heldout_colors
        self.split = not self.heldout
        self.init_RFD_dataloader()
        os.makedirs(self.filepath, exist_ok=True)


    def init_RFD_dataloader(self):
        def PILtoNumpy(img):
            return np.asarray(img)
        transform = torchvision.transforms.Lambda(PILtoNumpy)
        _set = RFDIterable(self.read_root, heldout_colors=self.heldout, transform=transform, shuffle=False) #in torch tensor format
        self.loader = DataLoader(_set, batch_size=self.chunksize, shuffle=False)

    @property
    def set_name(self) -> str:
        if self.heldout: return "HC"
        return ""

    def compute_files_partitions(self, length, num_files):
        num_chunks = (length//self.chunksize) #40
        chunks_per_file = num_chunks//num_files #4
        chunks_per_files = [chunks_per_file]*num_files
        if chunks_per_file*num_files*self.chunksize<length: #last file bigger
            chunks_per_files[-1] += ceil((length - chunks_per_file*num_files*self.chunksize)/self.chunksize)
        return chunks_per_files

    def _write_to_HDF5(self, filename:str, num_chunks:int):
        """ Test version of the write function
        num_chunks: number of chunks to read/write"""
        chunks_count = 1
        row_count = self.chunksize
        notify_every = 20
        updater = gen_bar_updater()
        # collect the first batch
        imgs, lbls = next(iter(self.loader))
        with h5py.File(self.filepath+filename, 'w') as f:
            #create the datasets
            images = f.create_dataset('images', shape=(num_chunks*imgs.shape[0],)+imgs.shape[1:],
                                      maxshape=(num_chunks*imgs.shape[0],)+imgs.shape[1:],
                                      chunks=imgs.shape, dtype=np.uint8, compression="gzip")
            images[:row_count] = imgs.cpu().detach().numpy()
            labels = f.create_dataset('labels', shape=(num_chunks*imgs.shape[0],)+lbls.shape[1:],
                                      maxshape=(num_chunks*imgs.shape[0],)+lbls.shape[1:],
                                      chunks=lbls.shape, dtype=np.uint8, compression="gzip")
            labels[:row_count] = lbls.cpu().detach().numpy()
            # iterate over the rest
            for imgs, lbls in self.loader:
                # Resize the dataset to accommodate the next chunk of rows
                next_row_count = row_count + imgs.shape[0]#less than chunksize if last batch
                # Write the next chunk
                images[row_count:next_row_count] = imgs.cpu().detach().numpy()
                labels[row_count:next_row_count] = lbls.cpu().detach().numpy()
                # Increment the row count
                row_count = next_row_count
                chunks_count+=1
                if chunks_count%notify_every==0:
                    updater(chunks_count, self.chunksize, min(num_chunks*self.chunksize, len(self.loader.dataset)))
                if chunks_count==num_chunks: return images.shape[0]

            return images.shape[0]

    def __call__(self, overwrite=False, **kwargs):
        num_files = kwargs.get('num_files',10)
        self.num_files = num_files
        base_name = f"RFD_{self.set_name}{kwargs.get('id', '')}_"
        length = kwargs.get('length', len(self.loader.dataset))
        print("Writing RFD dataset to .h5 file")

        if self.split and not kwargs.get("called", False):
            print("Splitting dataset into training and test")
            # Here we use recursion to avoid writing the same code twice
            # What happens: if we need to 'split' the dataset then we 'split' the 'call' to
            # first work on the training and then work on the test.
            # writing the training set
            self(overwrite=overwrite, num_files = int(self.train_test_split*num_files),
                 called=True, id="train", length=int(self.train_test_split*length))
            # writing the test set
            self(overwrite=overwrite, num_files = num_files - int(self.train_test_split*num_files),
                 called=True, id="test", length=length - int(self.train_test_split*length))
            return

        partitions = self.compute_files_partitions(length, num_files)
        for n in range(num_files):
            file_name = f"{base_name}{n}.h5"
            if os.path.exists(self.filepath+file_name):
                if not overwrite:
                    print(f"file {file_name} already there and overwrite=False. Nothing to do.")
                    break
                else:
                    print("Removing existing file.")
                    os.remove(self.filepath+file_name)
            print(f'Writing file number {n}')
            num_images_read = self._write_to_HDF5(file_name, partitions[n])
            print(f"{num_images_read} images read")



class RFDh5(torchvision.datasets.VisionDataset, DisentanglementDataset):
    """ Implementing RA dataset for RFD based on .h5 storage"""
    shape = (3, 128,128)
    raw_subfolders = ["finger","finger_heldout_colors","finger_real"]
    raw_files = [8, 2, 1] #train, test, heldout_test files

    def __init__(self,
                 root: str,
                 test: bool= False,
                 heldout_colors: bool =False,
                 real: bool=False,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        super().__init__(root)
        self.test = test
        self.heldout = heldout_colors
        self.real = real
        self.transform = transform
        self.target_transform = target_transform
        self.labels = None # to use for interventions
        self.size = None
        print(f"====== Opening RFD Dataset - {self.set_name} ======")
        self.info = self.read_info()
        if not self.real: self.partitions_dict = self.init_partitions() #open the partitions files
        else: self.real_set = self.read_real_images_and_labels()

    def close_all(self):
        #TODO: use this when finished to close all the hdf5 files opened
        pass


    def read_real_images_and_labels(self):
        """ Loading function only for real test dataset images"""
        images = np.load(self.raw_folder+"_images.npz", allow_pickle=True)["images"]
        labels = np.load(self.raw_folder+"_labels.npz", allow_pickle=True)["labels"]
        self.labels = labels
        return (images, labels)

    def get_labels(self):
        """ merge all the labels in a single numpy array so as to
        be able to uniquely identify the index of the item given the index of the label"""
        if self.labels is not None: return self.labels
        # not real set -> need to go through the partitions
        _labels = []
        for _, part in self.partitions_dict.items():
            _labels.append(part[0][1])
        self.labels = np.concatenate(_labels, axis=0)
        return self.labels


    def init_partitions(self):
        """Initialising the .h5 files and their order.
        Note: when reading .h5 file it won't be loaded into memory until its
        entries are called."""
        partitions_dict = {idx: self.open_partition(idx) for idx in range(self.num_files)}
        return partitions_dict

    def get_relative_index(self, index):
        """ Given absolute index it returns the partition index and the
        relative index inside the partition
        #TODO: make it work for multiple indices"""
        len_per_file = self.partitions_dict[0][1]
        partition_idx = index//len_per_file
        relative_index = index%len_per_file
        if index >= len_per_file*self.num_files:
            # contained in last partition
            relative_index = index - len_per_file*self.num_files
        return partition_idx, relative_index

    def open_partition(self, number):
        filename = f"RFD_{self.set_name}_{number}.h5"
        f = h5py.File(self.processed_folder + filename, 'r')
        images = f['images']
        labels = f['labels']
        length = images.shape[0]
        return (images, labels), length

    def __getitem__(self, index: int) -> Any:
        if self.real:
            imgs, lbls = self.real_set
            rel_idx = index
        else:
            par_idx, rel_idx = self.get_relative_index(index)
            imgs, lbls = self.partitions_dict[par_idx][0]
        img = torch.Tensor(imgs[rel_idx]/255.0) #rescaling the uint8
        img = img.permute(2,0,1) # from (H,W,C) to (C, H, W)
        lbl = torch.from_numpy(lbls[rel_idx]) #labels are integers
        # both the above are numpy arrays
        # so they need to be cast to torch tensors
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            lbl = self.target_transform(lbl)
        return img, lbl

    def __len__(self) -> int:
        #TODO: see here too for the "batched" case
        if self.heldout or self.real: return self.size
        if not self.test: return int(self.size*0.8)
        # only test is remaining
        return self.size -  int(self.size*0.8)

    def read_info(self):
        """ Opens the labels and info .npz files and stores them in the class"""
        # loading labels
        info = dict(np.load(self.raw_folder+"_info.npz", allow_pickle=True))
        self.size = info["dataset_size"].item()
        #TODO: extract interesting properties from the info here
        return info

    @property
    def num_factors(self):
        pass

    @property
    def factors_num_values(self):
        pass

    def sample_observations_from_factors(self, factors):
        pass

    @property
    def raw_folder(self) -> str:
        base = self.root + 'RFD/raw'
        if self.heldout: return str.join("/", [base, self.raw_subfolders[1], self.raw_subfolders[1]])
        if self.real: return str.join("/", [base, self.raw_subfolders[2], self.raw_subfolders[2]])
        return str.join("/", [base, self.raw_subfolders[0], self.raw_subfolders[0]])

    @property
    def num_files(self) -> int:
        if self.test: return self.raw_files[1]
        if self.heldout: return self.raw_files[2]
        return self.raw_files[0]

    @property
    def set_name(self) -> str:
        if self.test: return "test"
        if self.heldout: return "HC"
        return "train"

    @property
    def processed_folder(self) -> str:
        return self.root + 'RFD/processed/'




