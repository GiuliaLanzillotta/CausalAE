""" Script implementing logic for model visualisations """
import numpy as np
import os
import torch
from torchvision import utils as tvu
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from models import GenerativeAE
import itertools

class ModelVisualiser(object):
    """ Takes a trained model and an output directory and produces visualisations for
    the given model there. """
    def __init__(self, model:GenerativeAE, name, version, test_dataloader, **kwargs):
        super(ModelVisualiser, self).__init__()
        self.model = model
        self.name = name+"_"+version
        self.save_path = str.join("/",[".", kwargs.get("save_dir"), name, version])
        os.makedirs(self.save_path, exist_ok=True)
        self.test_dataloader = test_dataloader
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_reconstructions(self, current_epoch:int=None, grid_size:int=12, device=None):
        """ plots reconstructions from test set samples"""
        test_input, _ = iter(self.test_dataloader).__next__()
        with torch.no_grad():
            recons = self.model.generate(test_input.to(device))
        file_name = "reconstructions"
        if current_epoch is not None: file_name+="_"+str(current_epoch)
        tvu.save_image(recons.data,
                       fp= f"{self.save_path}/{file_name}.png",
                       normalize=True,
                       nrow=grid_size) # plot a square grid
        if current_epoch is not None and current_epoch==0: # print the originals
            tvu.save_image(test_input,
                           fp= f"{self.save_path}original.png",
                           normalize=True,
                           nrow=grid_size) # plot a square grid
        # clean
        del test_input, recons

    def plot_samples_from_prior(self, current_epoch:int=None, grid_size:int=12, device=None):
        """ samples from latent prior and plots reconstructions"""
        num_pics = grid_size**2 # square grid
        with torch.no_grad():
            random_codes = self.model.sample_noise_from_prior(num_pics) #sampling logic here
        recons = self.model.decode(random_codes.to(device))
        file_name = "prior_samples"
        if current_epoch is not None: file_name+="_"+str(current_epoch)
        tvu.save_image(recons.data,
                       fp= f"{self.save_path}/{file_name}.png",
                       normalize=True,
                       nrow=grid_size) # plot a square grid
        # clean
        del random_codes, recons

    def plot_latent_traversals(self, current_epoch:int=None, num_samples:int=5,
                               steps:int=26, device=None):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        test_input, _ = iter(self.test_dataloader).__next__()
        with torch.no_grad():
            noises = self.model.sample_noise_from_posterior(test_input[:num_samples].to(device))
        latents = noises.data.cpu().numpy()
        traversals = self.do_latent_traversal(latents, steps, width=0.2)
        latent_traversals = []
        for idx in range(latents.shape[0]): # for each sample
            vec = latents[idx,:]
            vec_base = np.stack([vec]*steps)
            # for each latent dimension, change the code in that dimension while keeping the others fixed
            vec_traversals = [list(vec_base + traversals[:,i].reshape(steps,1)) for i in range(latents.shape[1])]
            random_codes = np.stack(list(itertools.chain(*vec_traversals))) # (stepsxM) M-dimensional vectors
            recons = self.model.decode(torch.tensor(random_codes).to(device)) # (stepsxM) images
            latent_traversals.append(recons)
            file_name = "traversals_{}".format(idx)
            if current_epoch is not None: file_name+="_"+str(current_epoch)
            tvu.save_image(recons.data,
                           fp= f"{self.save_path}/{file_name}.png",
                           normalize=True,
                           nrow=steps) # steps x M grid

    @staticmethod
    def do_latent_traversal(latent_vectors, steps, width:float=0.5):
        # latent vectors is a NxM matrix, with M the number of latent dimensions
        # and N the number of samples
        minima = latent_vectors.min(0)
        maxima = latent_vectors.max(0)
        delta = maxima-minima
        traversals = np.linspace(-1*width*delta, delta*width, steps)
        # traversals has shape stepsxM
        return traversals

    def plot_training_gradients(self, current_epoch:int=None):
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
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.plot(max_grads, alpha=0.3, color="c")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.tick_params(axis='both', labelsize=4)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.tight_layout()
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        pic_name = "gradient"
        if current_epoch is not None: pic_name+="_"+str(current_epoch)
        plt.savefig(f"{self.save_path}/{pic_name}.png", dpi=200)