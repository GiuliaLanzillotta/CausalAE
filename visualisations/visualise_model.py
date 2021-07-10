""" Script implementing logic for model visualisations """
import numpy as np
import os
import torch
from torchvision import utils as tvu
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from models import GenerativeAE, ESAE, HybridLayer
import torchvision
import seaborn as sns
from torch.nn import functional as F

import itertools

class ModelVisualiser(object):
    """ Takes a trained model and a logger and produces visualisations, saving them to the logger."""
    def __init__(self, model:GenerativeAE,test_dataloader, **kwargs):
        super(ModelVisualiser, self).__init__()
        self.model = model
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_originals(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        num_plots = grid_size**2
        test_sample = self.test_input[:num_plots]
        grid_originals = torchvision.utils.make_grid(test_sample, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_originals.permute(1, 2, 0).cpu().numpy())
        return figure

    def plot_reconstructions(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        """ plots reconstructions from test set samples"""
        num_plots = grid_size**2
        test_sample = self.test_input[:num_plots]
        with torch.no_grad():
            recons = self.model.generate(test_sample.to(device), activate=True)
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        return figure

    def plot_samples_from_prior(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        """ samples from latent prior and plots reconstructions"""
        num_pics = grid_size**2 # square grid
        with torch.no_grad():
            random_codes = self.model.sample_noise_from_prior(num_pics) #sampling logic here
            recons = self.model.decode(random_codes.to(device), activate=True)
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        return figure

    def plot_samples_controlled_hybridisation(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        """ specific to ESAE model - to be used in ESAEmanager"""
        num_pics = grid_size**2 # square grid
        figures = []
        assert isinstance(self.model, ESAE), "The controlled hybridisation is only supported in ESAE models for now"
        with torch.no_grad():
            noises = self.model.controlled_sample_noise_from_prior(device=device, num_samples=num_pics) #sampling logic here
            for l,v in enumerate(noises):
                recons = self.model.decode(v.to(device), activate=True)
                grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
                figure = plt.figure(figsize=figsize)
                plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
                figures.append(figure)
        return figures

    def plot_latent_traversals(self, steps:int=10, device=None, figsize=(12,12), tailored=False, **kwargs):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        test_sample= self.test_input[11] #I like the number 11
        with torch.no_grad():
            code = self.model.encode_mu(test_sample.unsqueeze(0).to(device))
        latent_vector = code.data.cpu().numpy()
        dimensions = np.arange(latent_vector.shape[1])
        if not tailored: values = np.tile(np.linspace(-1., 1., num=steps), [dimensions, 1])
        else:
            #the values have to be tailored on the range of the specific dimensions
            #by default the selected range includes >= 90% of the density (with some approximation for hybrid layer)
            ranges = self.model.get_prior_range()
            values = np.stack([np.linspace(m,M, steps) for (m,M) in ranges],0)
        traversals = self.do_latent_traversals_multi_dim(latent_vector, dimensions, values, device=device)
        grid_traversals = torchvision.utils.make_grid(traversals, nrow=steps)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_traversals.permute(1, 2, 0).cpu().numpy())
        return figure

    def plot_hybridisation(self, Ni=10, Np=100, N=2, first=False, device=None, figsize=(12,12), **kwargs):
        """Takes Ni samples and hybridises them with Np samples, tracking the hybridisation and
        plotting the result.
        - N : number of dimensions that will undergo hybridisation """
        # extract samples
        all_samples = self.test_input[:Ni+Np]
        # encode
        with torch.no_grad():
            codes = self.model.encode_mu(all_samples.to(device))
        M = codes.shape[1]
        inputs = codes[:Ni]
        parents = codes[Ni:Ni+Np]
        # hybridise manually with hybrid layer
        try: unit_dim = self.model.unit_dim #SAE case
        except AttributeError: unit_dim=1 #VAE case
        new_vectors, parent_idx, to_resample = HybridLayer.hybridise_from_N(inputs, parents, N, unit_dim=unit_dim, first=first)
        # make an Nix(1+1+Np) grid plotting all the samples together with their parents
        complete_set = []
        for i,vec in enumerate(new_vectors):
            complete_set.append(vec.view(1,M))
            complete_set.append(inputs[i].view(1,M))
            complete_set.append(parents[parent_idx[i]])
        # this is a (2+N)*Ni long tensor
        complete_set = torch.cat(complete_set, dim=0)
        # decode and plot
        with torch.no_grad():
            recons = self.model.decode(complete_set.to(device), activate=True)
        grid_recons = torchvision.utils.make_grid(recons, nrow=N+2)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())

        return figure

    def plot_loss2distortion(self, device=None, **kwargs):
        """Plotting a curve measuring the amount of distortion along each dimension for
        multiple starting points"""
        figsize = kwargs.get("figsize",(20,20))
        N = kwargs.get("N",50)
        steps = kwargs.get("steps",21) # odd steps will include 0 in the linspace
        markersize = kwargs.get("markersize",20)
        font_scale = kwargs.get("font_scale",20)
        ylim = kwargs.get("ylim")
        xlim = kwargs.get("xlim")

        idx = torch.randperm(self.test_input.shape[0])[:N]
        base_vecs = self.test_input[idx]
        # encode - apply distortion - decode
        with torch.no_grad():
            codes = self.model.encode_mu(base_vecs.to(device))
        ranges = self.model.get_prior_range() # (min, max) for each dimension
        widths = [M-m for (m,M) in ranges]
        distortion_levels = [np.linspace(-w,+w,steps) for w in widths]
        ys = []
        for i in range(N):
            # losses is a list of floats
            losses = self.do_latent_distortion_multi_dim(codes[i], distortion_levels, base_vecs[i], device=device)
            ys.append(losses)
        ys = torch.stack(ys).cpu().numpy() # N x D x steps
        # now we want to plot N functions relatin the distortion levels to the different ys for each dimension
        D = len(distortion_levels)
        nrows = kwargs.get("nrows", D//3); ncols = D//nrows +1 if D%nrows!=0 else D//nrows
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(nrows, ncols, sharey=True, figsize=figsize)
        dim=0
        for row in range(nrows):
            for col in range(ncols):
                if dim==D: break
                Nlosses = ys[:,dim,:] # N x steps
                ax[row,col].set_title(f"Latent dimension {dim}")
                axi = sns.lineplot(np.tile(distortion_levels[dim], N), Nlosses.reshape(-1,), ax=ax[row,col],  marker=".", markersize=markersize)
                ax[row,col].axvline(0, color='r', linestyle="--")
                axi.set(ylabel='L1 distance on pixels', xlabel='Latent space distortion')
                axi.tick_params(axis="x", labelsize=font_scale)
                axi.tick_params(axis="y", labelsize=font_scale)
                if ylim is not None: axi.set(ylim=(0, ylim))
                if xlim is not None: axi.set(xlim=(-xlim, xlim))
                dim+=1

        return fig








    def do_latent_distortion_multi_dim(self, latent_vector, distortion_levels, original_sample, device=None):
        """ For each dimension in the original latent vector applies a sliding level of distortion
        Returns tensor of size (num_dimensions) x num_steps - where num_steps is the number of distortion levels
        applied to each dimension"""
        losses = []
        num_values = distortion_levels[0].shape[0]

        for i, dist in enumerate(distortion_levels):
            # Creates num_values copy of the latent_vector along the first axis.
            _vectors = np.tile(latent_vector.cpu().numpy(), [num_values, 1])
            # Intervenes in the latent space.
            _vectors[:, i] = dist
            # Generate the batch of images and computes the loss as MSE distance
            with torch.no_grad():
                recons = self.model.decode(torch.tensor(_vectors, dtype=torch.float).to(device), activate=True)
                # recons has shape stepsx(inout shape)
            diff = recons - original_sample.to(device)
            loss = torch.norm(diff, 1, dim=tuple(range(diff.dim())[1:]))
            losses.append(loss)
        return torch.vstack(losses)



    def do_latent_traversals_multi_dim(self, latent_vector, dimensions, values, device=None):
        """ Creates a tensor where each element is obtained by passing a
            modified version of latent_vector to the generato. For each
            dimension of latent_vector the value is replaced by a range of
            values, obtaining len(values) different elements.

            values is an array with shape [num_dimensionsXsteps]

        latent_vector, dimensions, values are all numpy arrays
        """
        num_values = values.shape[1]
        traversals = []
        for dimension, _values in zip(dimensions, values):
            # Creates num_values copy of the latent_vector along the first axis.
            latent_traversal_vectors = np.tile(latent_vector, [num_values, 1])
            # Intervenes in the latent space.
            latent_traversal_vectors[:, dimension] = _values
            # Generate the batch of images
            with torch.no_grad():
                images = self.model.decode(torch.tensor(latent_traversal_vectors, dtype=torch.float).to(device), activate=True)
                # images has shape stepsx(image shape)
            traversals.append(images)
        return torch.cat(traversals, dim=0)

    def plot_training_gradients(self, figsize=(12,12)):
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
                    ave_grads.append(p.grad.cpu().abs().mean())
                    max_grads.append(p.grad.cpu().abs().max())
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
        return figure