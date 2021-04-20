""" Script implementing logic for model visualisations """
import numpy as np
import os
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
        recons = self.model.forward(test_input.to(self.device))
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
        # random_codes = ... sampling logic here
        # recons = model() ...

    def plot_latent_traversals(self, current_epoch:int=None, grid_size:int=12):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        # for each latent dimension, change the code in that dimension while keeping the others fixed
        pass

    def plot_training_gradients(self, current_epoch:int=None):
        pass