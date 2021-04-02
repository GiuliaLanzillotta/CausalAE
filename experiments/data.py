# Script responsible for data loading and cleaning
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, CelebA
import matplotlib.pyplot as plt
import torch

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

        if args["dataset_name"] == 'cifar10':
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
            data_folder = '/datasets/celeba/'
            train_set = CelebA(data_folder,
                               split='train',
                               download=True,
                               transform=transform)
            test_set = CelebA(data_folder,
                              split='valid',
                              download=True,
                              transform=transform)
        
        else:
            raise RuntimeError("Unrecognized data set '{}'".format(
                args.dataset_name))

        #TODO: add validation set
        #train, val = torch.utils.data.random_split(train_set, lengths=[55000, 5000], generator=torch.Generator().manual_seed(42))

        self.train = DataLoader(train_set,
                                batch_size=args["batch_size"],
                                shuffle=True,
                                drop_last=True)
        self.test = DataLoader(test_set,
                                batch_size=args["test_batch_size"],
                                shuffle=False)

        self.data_shape = self.train.dataset[0][0].size()
        self.img_size = self.data_shape[1:]
        self.color_ch = self.data_shape[0]
    
    def plot_images(self, images, cls_true, cls_pred=None):
        """Plot 9 sample images in a 3x3 sub-plot."""
        assert len(images) == len(cls_true) == 9
        # Convert list of torch.Tensor to list numpy.array and 
        # rearrange dimensions from (C,H,W) to (H,W,C)
        images = [tensor.numpy().transpose() for tensor in images]
        # Create figure with 3x3 sub-plots.
        fig, axes = plt.subplots(3, 3)
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        for i,ax in enumerate(axes.flat):
            ax.imshow(images[i].reshape(self.data_shape))
        # Show true and predicted classes.
        xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i]) if cls_pred else "True: {0}".format(cls_true[i])
        ax.set_xlabel(xlabel)
        # Remove ticks 
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_from_test(self):
        """ Take first 9 images from test set and plot them
            #TODO: random extraction 
        """
        images, cls_true = zip(*[self.test.dataset[i] for i in range(9)])
        plot_images(images=images, cls_true=cls_true)