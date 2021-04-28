# Script responsible for data loading and cleaning
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR10, SVHN, CelebA, FashionMNIST
import matplotlib.pyplot as plt
from datasets import RFD, Shapes3d
import numpy as np
import torch
import os

class DatasetLoader:
    """
    Wrapper for DataLoaders. 
    Data attributes:
    - train: DataLoader object for training set
    - test: DataLoader object for test set
    - data_shape: shape of each data point (channels, height, width)
    - img_size: spatial dimensions of each data point (height, width)
    - color_ch: number of color channels
    """

    def __init__(self, args):
        """ 
        args - dict containing:
           - dataset_name -> supported: ['cifar10', 'svhn', 'celeba']
           - batch_size (training set batch size )
           - test_batch_size 
        $note: to obtain a dict from a Namespace simply >> vars(nsObject)
        """
        already_split = False
        if args["dataset_name"] == 'MNIST':
            transform = transforms.ToTensor()
            root = os.path.dirname(os.path.realpath(__file__))
            data_folder = os.path.join(root,'../datasets/MNIST/')
            train_set = MNIST(data_folder,
                                train=True,
                                download=True,
                                transform=transform)
            test_set = MNIST(data_folder,
                               train=False,
                               download=True,
                               transform=transform)

        elif args["dataset_name"] == 'FashionMNIST':
            transform = transforms.ToTensor()
            root = os.path.dirname(os.path.realpath(__file__))
            data_folder = os.path.join(root,'../datasets/FashionMNIST/')
            train_set = FashionMNIST(data_folder,
                                     train=True,
                                     download=True,
                                     transform=transform)
            test_set = FashionMNIST(data_folder,
                                    train=False,
                                    download=True,
                                    transform=transform)

        elif args["dataset_name"] == 'cifar10':
            transform = transforms.ToTensor()
            data_folder = './datasets/cifar10/'
            train_set = CIFAR10(data_folder,
                                train=True,
                                download=True,
                                transform=transform)
            test_set = CIFAR10(data_folder,
                               train=False,
                               download=True,
                               transform=transform)

        elif args["dataset_name"] == 'svhn':
            transform = transforms.ToTensor()
            data_folder = './datasets/svhn/'
            train_set = SVHN(data_folder,
                             split='train',
                             download=True,
                             transform=transform)
            test_set = SVHN(data_folder,
                            split='test',
                            download=True,
                            transform=transform)

        elif args["dataset_name"] == 'celeba':
            transform = transforms.Compose([
                transforms.CenterCrop(148),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
            ])
            data_folder = './datasets/celeba/'
            train_set = CelebA(data_folder,
                               split='train',
                               download=True,
                               transform=transform)
            valid_set = CelebA(data_folder,
                              split='valid',
                              download=True,
                              transform=transform)
            test_set = CelebA(data_folder,
                              split='test',
                              download=True,
                              transform=transform)
            already_split = True
            tot = len(train_set) + len(valid_set) + len(test_set)
            self.num_samples = tot


        elif args["dataset_name"] == 'RFD': #new dataset: https://arxiv.org/pdf/2010.14407.pdf
            transform = transforms.ToTensor()
            cluster_data_folder = '/cluster/scratch/glanzillo/robot_finger_datasets/'
            standard_set = RFD(cluster_data_folder,
                            transform=transform)
            heldout_set = RFD(cluster_data_folder,
                              heldout_colors=True,
                              transform=transform)
            real_set = RFD(cluster_data_folder,
                           real=True,
                           transform=transform)
            # Note that these two additional test sets are only available with the RFD dataset
            self.heldout_set = DataLoader(heldout_set,
                                          batch_size=args["test_batch_size"],
                                          shuffle=False)
            self.real_set = DataLoader(real_set,
                                       batch_size=args["test_batch_size"],
                                       shuffle=False)
            # train, val, test set for standard set (to be used during training)
            # note that the split sizes are fixed here (70,20,10 split)
            tot = standard_set.size
            self.num_samples = tot
            train_num = int(0.7*tot)
            val_num = int(0.2*tot)
            test_num = tot-train_num-val_num
            train_set, valid_set, test_set = torch.utils.data.random_split(standard_set,
                                                             lengths=[train_num, val_num, test_num],
                                                             generator=torch.Generator().manual_seed(42))
            already_split = True

        elif args["dataset_name"] == '3DShapes':
            transform = transforms.ToTensor()
            data_folder = './datasets/Shapes3d'
            dataset = Shapes3d(data_folder,
                               transform=transform,
                               download=True)
            # train, val, test set (to be used during training)
            # note that the split sizes are fixed here (70,20,10 split)
            tot = len(dataset)
            self.num_samples = tot
            train_num = int(0.7*tot)
            val_num = int(0.2*tot)
            test_num = tot-train_num-val_num
            train_set, valid_set, test_set = torch.utils.data.random_split(dataset,
                                                                           lengths=[train_num, val_num, test_num],
                                                                           generator=torch.Generator().manual_seed(42))
            already_split = True

        else:
            raise RuntimeError("Unrecognized data set '{}'".format(
                args.dataset_name))

        if not already_split:
            tot_train = train_set.data.shape[0]
            train_num = int(0.7*tot_train)
            val_num = tot_train-train_num
            train_set, valid_set = torch.utils.data.random_split(train_set,
                                                                 lengths=[train_num, val_num],
                                                                 generator=torch.Generator().manual_seed(42))
            self.num_samples = tot_train + test_set.data.shape[0]

        self.train = DataLoader(train_set,
                                batch_size=args["batch_size"],
                                shuffle=False,
                                drop_last=True,
                                num_workers=8)
        self.val = DataLoader(valid_set,
                              batch_size=args["batch_size"],
                              shuffle=False,
                              drop_last=True,
                              num_workers=2)
        self.test = DataLoader(test_set,
                               batch_size=args["test_batch_size"],
                               shuffle=False,
                               num_workers=2)


        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]

    
    @staticmethod
    def plot_images(images, cls_true, cls_pred=None,
                    num_images:int=9):
        """Plot 9 sample images in a 3x3 sub-plot."""
        assert len(images) == len(cls_true) == num_images
        ncols = int(np.ceil(num_images**0.5))
        nrows = int(np.ceil(num_images / ncols))
        # Convert list of torch.Tensor to list numpy.array and 
        # rearrange dimensions from (C,H,W) to (H,W,C)
        try: images = [tensor.numpy().transpose() for tensor in images]
        except: pass
        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(ncols, nrows, figsize=(nrows * 3, ncols * 3))
        axes = axes.flatten()
        fig.subplots_adjust(hspace=0.3, wspace=0.3)

        for i,ax in enumerate(axes.flat):
            if i < num_images:
                ax.imshow(images[i], cmap='Greys_r', interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                # Show true and predicted classes.
                xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i]) if cls_pred else "True: {0}".format(cls_true[i])
                ax.set_xlabel(xlabel)
            else:
                ax.axis('off')

    def plot_from_test(self, num_samples:int=9):
        """ Take first 9 images from test set and plot them
        """
        indices = np.random.choice(len(self.test), num_samples)
        images, cls_true = zip(*[self.test.dataset[i] for i in indices])
        self.plot_images(images=images, cls_true=cls_true)