""" Script implementing logic for model visualisations """
import numpy as np
import os
import torch
from torchvision import utils as tvu
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from models import GenerativeAE
import torchvision
import itertools

class ModelVisualiser(object):
    """ Takes a trained model and an output directory and produces visualisations for
    the given model there. """
    def __init__(self, model:GenerativeAE,test_dataloader, **kwargs):
        super(ModelVisualiser, self).__init__()
        self.model = model
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_reconstructions(self, logger, global_step:int=None, grid_size:int=12, device=None, figsize=(12,12)):
        """ plots reconstructions from test set samples"""
        num_plots = grid_size**2
        test_sample = self.test_input[:num_plots]
        with torch.no_grad():
            recons = self.model.generate(test_sample.to(device), activate=True)
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        logger.add_figure("reconstructions", figure, global_step=global_step)
        if global_step is not None and global_step==0: # print the originals
            grid_originals = torchvision.utils.make_grid(test_sample, nrow=grid_size)
            figure = plt.figure(figsize=figsize)
            plt.imshow(grid_originals.permute(1, 2, 0).cpu().numpy())
            logger.add_figure("originals", figure)
        # clean
        del recons

    def plot_samples_from_prior(self, logger, global_step:int=None, grid_size:int=12, device=None, figsize=(12,12)):
        """ samples from latent prior and plots reconstructions"""
        num_pics = grid_size**2 # square grid
        with torch.no_grad():
            random_codes = self.model.sample_noise_from_prior(num_pics) #sampling logic here
        recons = self.model.decode(random_codes.to(device), activate=True)
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        logger.add_figure("prior_samples", figure, global_step=global_step)
        # clean
        del random_codes, recons

    def plot_latent_traversals(self, logger, global_step:int=None, num_samples:int=1,
                               steps:int=10, device=None, figsize=(12,12)):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        for idx in np.random.randint(0,self.test_input.shape[0],num_samples): # for each sample
            test_sample= self.test_input[idx]
            with torch.no_grad():
                codes = self.model.encode(test_sample.unsqueeze(0).to(device))
            try: codes = codes[1] # if the output is a list extract the second element (VAE case)
            except: pass
            latents = codes.data.cpu().numpy()
            traversals = np.linspace(-3.0, 3.0, steps) #todo: make this grid more flexible
            base = np.stack([latents]*steps).squeeze(1)
            # for each latent dimension, change the code in that dimension while keeping the others fixed
            vec_traversals = [list(np.hstack([base[:,:i], traversals.reshape(steps,1), base[:,i+1:]])) for i in range(latents.shape[1])]
            random_codes = np.stack(list(itertools.chain(*vec_traversals))) # (stepsxM) M-dimensional vectors
            with torch.no_grad():
                recons = self.model.decode(torch.tensor(random_codes, dtype=torch.float).to(device), activate=True) # (stepsxM) images
            grid_recons = torchvision.utils.make_grid(recons, nrow=steps)
            figure = plt.figure(figsize=figsize)
            plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
            logger.add_figure("traversals", figure, global_step=global_step)

    def plot_training_gradients(self, logger, global_step:int=None, figsize=(12,12)):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in self.model.named_parameters():
            if(p.requires_grad) and ("bias" not in n):
                try:
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                    layers.append(n)
                except AttributeError:
                    print(n+" has no gradient. Skipping")
                    continue
        figure = plt.figure(figsize=figsize)
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.plot(max_grads, alpha=0.3, color="c")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.tick_params(axis='both', labelsize=4)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.002) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.tight_layout()
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        logger.add_figure("gradient", figure, global_step=global_step)