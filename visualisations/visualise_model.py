""" Script implementing logic for model visualisations """
import numpy as np
import os
import torch
from torchvision import utils as tvu
from models import GenerativeAE

class ModelVisualiser(object):
    """ Takes a trained model and an output directory and produces visualisations for
    the given model there. """
    def __init__(self, model:GenerativeAE, name, version, test_dataloader, device, **kwargs):
        super(ModelVisualiser, self).__init__()
        self.model = model
        self.name = name+"_"+version
        self.save_path = str.join("/",[".", kwargs.get("save_dir"), name, version])
        os.makedirs(self.save_path, exist_ok=True)
        self.test_dataloader = test_dataloader
        self.device = device # the device where the model is
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_reconstructions(self, current_epoch:int=None, grid_size:int=12):
        """ plots reconstructions from test set samples"""
        test_input, _ = next(iter(self.test_dataloader()))
        with torch.no_grad():
            recons = self.model.generate(test_input.to(self.device))
        file_name = "reconstructions"
        if current_epoch is not None: file_name+="_"+str(current_epoch)
        tvu.save_image(recons.data,
                       fp= f"{self.save_path}/{file_name}.png",
                       normalize=True,
                       nrow=grid_size) # plot a square grid
        # clean
        del test_input, recons

    def plot_samples_from_prior(self, current_epoch:int=None, grid_size:int=12):
        """ samples from latent prior and plots reconstructions"""
        num_pics = grid_size**2 # square grid
        with torch.no_grad():
            random_codes = self.model.sample_noise_from_prior(num_pics) #sampling logic here
        recons = self.model.decode(random_codes)
        file_name = "prior_samples"
        if current_epoch is not None: file_name+="_"+str(current_epoch)
        tvu.save_image(recons.data,
                       fp= f"{self.save_path}/{file_name}.png",
                       normalize=True,
                       nrow=grid_size) # plot a square grid
        # clean
        del random_codes, recons

    def plot_latent_traversals(self, current_epoch:int=None, num_samples:int=10, steps:int=12):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        test_input, _ = next(iter(self.test_dataloader()))
        with torch.no_grad():
            noises = self.model.sample_noise_from_posterior(test_input[:num_samples])
        traversals = self.do_latent_traversal(noises.data.numpy(), steps, tolerance=0.01)
        # for each latent dimension, change the code in that dimension while keeping the others fixed
        for idx in range(num_samples):
            noise = noises.data.numpy[idx,:]
            traversals_idx = [noise]*steps
            #todo: finish here

    @staticmethod
    def do_latent_traversal(latent_vectors, steps, tolerance:float=0.01):
        # latent vectors is a NxM matrix, with M the number of latent dimensions
        # and N the number of samples
        minima = latent_vectors.min(0)
        maxima = latent_vectors.max(0)
        delta = maxima-minima
        traversals = np.linspace(minima-tolerance*delta, maxima+tolerance*delta, steps)
        # traversals has shape stepsxM
        return traversals

    def plot_training_gradients(self, current_epoch:int=None):
        pass