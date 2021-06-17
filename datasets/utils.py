""" A few helpful functions when dealing with data storage"""
from torch.utils.model_zoo import tqdm
from typing import  Callable
from torch import Tensor
import torch
import numpy as np

def gen_bar_updater() -> Callable[[int, int, int], None]:
    pbar = tqdm(total=None)

    def bar_update(count, block_size, total_size):
        if pbar.total is None and total_size:
            pbar.total = total_size
        progress_bytes = count * block_size
        pbar.update(progress_bytes - pbar.n)

    return bar_update

class AdditiveNoise(object):
    """ Implementation of additive Gaussin noise to be used as a transformation in the data processing pipeline.
    Adding noise to the input of neural networks has been shown to be beneficial for out-of-distribution
    generalization (Sietsma & Dow, 1991; Bishop, 1995)."""
    smooth_noise_level=0.15
    fine_noise_level=0.03

    def __init__(self, factor:int=16):
        self.smooth_noise_factor =factor
        self.upsampling = torch.nn.UpsamplingBilinear2d(scale_factor=self.smooth_noise_factor)
        print("Additive noise initialised.")


    def __call__(self, pic:Tensor)->Tensor:
        shape = pic.shape
        noise_width = shape[2]//self.smooth_noise_factor

        # Noise one:  8 × 8 pixel-wise (greyscale) noise with standard deviation 0.15, bilinearly upsampled by a
        # factor of 16.
        noise1 = torch.randn((1, 1, noise_width, noise_width))*self.smooth_noise_level
        noise1 = self.upsampling(noise1)
        noise1 = torch.squeeze(noise1, dim=0) #bilinear upsampling needs 4 dimensional tensor
        pic+= noise1

        # Noise two: independent for each subpixel (RGB)
        noise2 = torch.randn_like(pic)*self.fine_noise_level
        pic+= noise2
        pic = torch.clamp(pic, 0.0, 1.0)
        return pic

    def __repr__(self):
        return self.__class__.__name__ + "()"

def transform_discrete_labels(labels, range_per_factor):
    """Performs transformation of discrete labels into numpy categoricals
    Given a numpy array ov labels of any type returns an array of ints
    Labels is of shape (num_samples, num_factors)
    num_values_per_factor: array (num_factors,1) with number of categories per variable
    range_per_factor: list of bins to use for each factor"""
    discretized = np.zeros_like(labels, dtype=np.int)
    for i in range(labels.shape[1]):
        discretized[:, i] = np.digitize(labels[:, i], bins=range_per_factor[i])
    return discretized