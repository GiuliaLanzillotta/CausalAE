""" A few helpful functions when dealing with data storage"""
from torch.utils.model_zoo import tqdm
from typing import  Callable
from torch import Tensor
import torch

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
    smooth_noise_level=0.15,
    smooth_noise_factor=16,
    fine_noise_level=0.03

    def __init__(self):
        self.upsampling = torch.nn.UpsamplingBilinear2d(size=(self.smooth_noise_factor, self.smooth_noise_factor))


    def __call__(self, pic:Tensor)->Tensor:
        shape = pic.shape
        noise_width = shape[2]//self.smooth_noise_factor

        # Noise one:  8 × 8 pixel-wise (greyscale) noise with standard deviation 0.15, bilinearly upsampled by a
        # factor of 16.
        noise1 = torch.randn((shape[0], 1, noise_width, noise_width))*self.smooth_noise_level
        noise1 = self.upsampling(noise1)
        pic+= noise1 #TODO: error here - The size of tensor a (16) must match the size of tensor b
        # (128) at non-singleton dimension 3

        # Noise two: independent for each subpixel (RGB)
        noise2 = torch.randn_like(pic)*self.fine_noise_level
        pic+= noise2
        pic = torch.clip(pic, 0.0, 1.0)
        return pic
        """
        shp = imgs.get_shape()
        shp = tf.TensorShape([
            shp[0], shp[1] // smooth_noise_factor, shp[2] // smooth_noise_factor,
            tf.Dimension(1)
        ])
        noise = tf.random.normal(shp, stddev=smooth_noise_level)
        noise = upsample2d(noise, size=(smooth_noise_factor, smooth_noise_factor))
    
        imgs += noise
    
        noise = tf.random.normal(imgs.shape, stddev=fine_noise_level)
        imgs += noise
    
        # img = np.clip(img, a_min=0.0, a_max=1.0)
        imgs = tf.clip_by_value(imgs, clip_value_min=0.0, clip_value_max=1.0)
        """

    def __repr__(self):
        pass