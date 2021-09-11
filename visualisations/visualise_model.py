""" Script implementing logic for model visualisations """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchvision
from matplotlib.lines import Line2D
from torch import Tensor
from . import utils
from .vis_responses import traversal_responses
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


from models import GenerativeAE, ESAE, HybridLayer


class ModelVisualiser(object):
    """ Takes a trained model and a logger and produces visualisations, saving them to the logger."""
    def __init__(self, model:GenerativeAE,test_dataloader, **kwargs):
        super(ModelVisualiser, self).__init__()
        self.model = model
        self.dataloader = test_dataloader
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)

    def plot_originals(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        num_plots = grid_size**2
        test_sample = self.test_input[:num_plots]
        grid_originals = torchvision.utils.make_grid(test_sample, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_originals.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.grid(b=None)
        return figure

    def plot_reconstructions(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        """ plots reconstructions from test set samples"""
        num_plots = grid_size**2
        test_sample = self.test_input[:num_plots]
        with torch.no_grad():
            recons = self.model.reconstruct(test_sample.to(device), activate=True).detach()
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.grid(b=None)
        return figure

    def plot_samples_from_prior(self, grid_size:int=12, device=None, figsize=(12,12), **kwargs):
        """ samples from latent prior and plots reconstructions"""
        num_pics = grid_size**2 # square grid
        with torch.no_grad():
            recons = self.model.generate(num_pics, activate=True, device=device).detach()
        grid_recons = torchvision.utils.make_grid(recons, nrow=grid_size)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.grid(b=None)
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
                plt.axis('off')
                plt.grid(b=None)
                figures.append(figure)
        return figures

    def plot_latent_traversals(self, steps:int=10, device=None, figsize=(12,12), tailored=False, **kwargs):
        """ traverses the latent space in different axis-aligned directions and plots
        model reconstructions"""
        # get posterior codes
        test_sample= self.test_input[11] #I like the number 11
        with torch.no_grad():
            code = self.model.encode_mu(test_sample.unsqueeze(0).to(device), update_prior=False).detach()
        latent_vector = code.cpu().numpy()
        dimensions = np.arange(latent_vector.shape[1])
        if not tailored: ranges = [(-1.,1.)]*len(dimensions)
        else: ranges = self.model.get_prior_range()
        values = utils.get_traversals_steps(steps, ranges).cpu().numpy()
        traversals = utils.do_latent_traversals_multi_dim(self.model, latent_vector, dimensions, values, device=device)
        grid_traversals = torchvision.utils.make_grid(traversals, nrow=steps)
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_traversals.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.grid(b=None)
        return figure

    def plot_hybridisation(self, Np=100, device=None, figsize=(12,12), **kwargs):
        """Takes Ni samples and hybridises them with Np samples, tracking the hybridisation and
        plotting the result.
        - N : number of dimensions that will undergo hybridisation
        - Ni : number of times the
        - Np : size of parents samples set to draw from """
        # extract samples
        seed = kwargs.get("seed",11)
        rng = np.random.RandomState(seed)
        all_samples = self.test_input[:1+Np]
        # encode
        with torch.no_grad():
            codes = self.model.encode_mu(all_samples.to(device), update_prior=False).detach()
        M = codes.shape[1]
        base_idx = rng.randint(1+Np)
        base = codes[base_idx].view(1,-1)
        parents = torch.vstack([codes[:base_idx], codes[base_idx+1:]])
        # hybridise manually with hybrid layer
        try: unit_dim = self.model.unit_dim #SAE case
        except AttributeError: unit_dim=1 #VAE case
        complete_set = []
        for u in range(M//unit_dim):
            # two M dimensional vectors
            new_vector, parent_idx = HybridLayer.hybridise_from_N(base, parents, [u], unit_dim=unit_dim, random_state=rng)
            # 1x(3) grid plotting all the samples together with their parents
            complete_set.append(new_vector.view(1,M).detach())
            complete_set.append(base.view(1,M).detach())
            complete_set.append(parents[parent_idx].view(1,M).detach())

        # this is a (3*M) long tensor
        complete_set = torch.cat(complete_set, dim=0)
        # decode and plot
        with torch.no_grad():
            recons = self.model.decode(complete_set.to(device), activate=True).detach()
        grid_recons = torchvision.utils.make_grid(recons, nrow=3) #nrow is actually ncol
        figure = plt.figure(figsize=figsize)
        plt.imshow(grid_recons.permute(1, 2, 0).cpu().numpy())
        plt.axis('off')
        plt.grid(b=None)

        return figure

    def plot_marginal(self, device=None, **kwargs):
        """plotting marginal distributions over latent dimensions"""
        figsize = kwargs.get("figsize",(20,20))
        num_batches = kwargs.get("num_batches",10)
        n_bins = kwargs.get("bins",200)
        pair = kwargs.get("pair",False) # whether to make a pairplot over the latents
        xlim = kwargs.get("xlim")
        font_scale = kwargs.get("font_scale",20)

        _all_codes = []
        for b in range(num_batches):
            batch, _ = next(iter(self.dataloader))
            with torch.no_grad():
                codes = self.model.encode_mu(batch.to(device), update_prior=True, integrate=True)
                _all_codes.append(codes)
        _all_codes = torch.vstack(_all_codes) # (B x num batches) x D

        N, D = _all_codes.shape
        if not pair:
            nrows = 3; ncols = D//nrows +1 if D%nrows!=0 else D//nrows
            fig, ax = plt.subplots(nrows, ncols, sharey="all", figsize=figsize)
            dim=0
            for row in range(nrows):
                for col in range(ncols):
                    if dim==D: break
                    axi = sns.histplot(_all_codes[:,dim].cpu().numpy(), ax=ax[row,col], kde=True, bins=n_bins, fill=False)
                    axi.set(ylabel='Number of observations', xlabel=f'Latent dim {dim}')
                    axi.tick_params(axis="x", labelsize=font_scale)
                    axi.tick_params(axis="y", labelsize=font_scale)
                    if xlim is not None: axi.set(xlim=(-xlim, xlim))
                    dim+=1
            return fig

        fig = plt.figure(figsize=figsize)
        axi = sns.pairplot(pd.DataFrame(_all_codes.cpu().numpy()), diag_kws = {'alpha':0.55, 'bins':200, 'kde':True})
        return fig

    #TODO: move distortion functions to dedicated module

    def _compute_output_distortion(self, latents, originals, device):
        """Computes loss on the pixel space for the given latents
        """
        with torch.no_grad():
            recons = self.model.decode(latents.to(device), activate=True) # N x image_dim
            diff = recons - originals.to(device) # N x image_dim
            loss = torch.norm(diff, 1, dim=tuple(range(diff.dim())[1:])) # N x 1
        return loss

    def _compute_max_loss(self, codes:Tensor, originals:Tensor, eps:float, dim:int, device):
        """Computes value of maximal loss for an epsilon sized distortion on the dimension dim"""
        altered_codes = codes.clone()
        # positive distortion
        altered_codes[:,dim] += eps
        losses = self._compute_output_distortion(altered_codes, originals, device)
        # negative distortion
        altered_codes = codes.clone()
        altered_codes[:,dim] -= eps
        new_losses = self._compute_output_distortion(altered_codes, originals, device)
        losses = torch.max(torch.stack([losses, new_losses]), dim=0)[0] # N x 1
        return losses # N x 1 tensor

    def plot_loss2marginaldistortion(self, device="cpu", figsize=(12,12), **kwargs):
        """Given a fixed distortion size the plot shows the amount of error increase in the output
        given by applying the distortion to the latent space - basically showing the effect of an epsilon-sized L1
        adversarial attack on each latent dimension"""
        N = kwargs.get("N",50)
        relative = kwargs.get("relative",False) # whether to print only the difference from the base error or both base and incurred error
        ro = kwargs.get("ro",0.1) # magnitude of distortion (in percentage) - it varies for each dimension depending on its range
        markersize = kwargs.get("markersize",20)
        font_scale = kwargs.get("font_scale",20)
        ylim = kwargs.get("ylim")
        xlim = kwargs.get("xlim")

        idx = torch.randperm(self.test_input.shape[0])[:N]
        base_vecs = self.test_input[idx]
        # encode - apply distortion - decode
        with torch.no_grad():
            codes = self.model.encode_mu(base_vecs.to(device), update_prior=True, integrate=True)
            ranges = self.model.get_prior_range() # (min, max) for each dimension

        D = len(ranges)
        distortions = [ro*(M-m) for (m,M) in ranges] # D x 1
        ys = []
        for i in range(D):
            # losses is a list of floats
            losses = self._compute_max_loss(codes, base_vecs, eps=distortions[i], dim=i, device=device)
            ys.append(losses)
        ys = torch.stack(ys).cpu().numpy() # D x N
        initial_loss = self._compute_output_distortion(codes, base_vecs, device).cpu().numpy() # N x 1


        nrows = kwargs.get("nrows", D//3); ncols = D//nrows +1 if D%nrows!=0 else D//nrows
        sns.set_style('darkgrid')
        fig, ax = plt.subplots(nrows, ncols, sharey='all', figsize=figsize)
        dim=0
        for row in range(nrows):
            for col in range(ncols):
                if dim==D: break
                ax[row,col].set_title(f"Distortion of {distortions[dim]:.5f} applied")
                x = codes[:,dim].cpu().numpy() # N x 1
                Nlosses = ys[dim,:] # N x 1
                hues = None
                if not relative:
                    y = np.hstack([Nlosses,initial_loss])
                    x = np.tile(x,2)
                    hues = [i for i in range(2) for _ in range(N)] # [1,1,1,...,1,2,2,2,...,2]
                else:
                    # relative increase
                    y = (Nlosses - initial_loss)/distortions[dim]
                axi = sns.lineplot(x, y, ax=ax[row,col], marker=".", markersize=markersize, hue=hues, palette="Blues")
                if not relative: axi.legend(labels=['+ distortion', 'base loss'])
                ax[row,col].axvline(np.mean(x), color='r', linestyle="--")
                axi.set(ylabel='Worst L1 distance increase for fixed latent distortion',
                        xlabel=f'Latent dim {dim}')
                axi.tick_params(axis="x", labelsize=font_scale)
                axi.tick_params(axis="y", labelsize=font_scale)
                if ylim is not None: axi.set(ylim=(0, ylim))
                if xlim is not None: axi.set(xlim=(-xlim, xlim))
                dim+=1

        return fig

    def plot_loss2distortion(self, device=None, **kwargs):
        """Plotting a curve measuring the amount of distortion along each dimension for
        multiple starting points"""
        figsize = kwargs.get("figsize",(20,30))
        N = kwargs.get("N",50)
        steps = kwargs.get("steps",21) # odd steps will include 0 in the linspace
        markersize = kwargs.get("markersize",20)
        font_scale = kwargs.get("font_scale",20)
        x_scale = kwargs.get("x_scale",1.0) # inflating the size of the latent
        ylim = kwargs.get("ylim")
        xlim = kwargs.get("xlim")

        idx = torch.randperm(self.test_input.shape[0])[:N]
        base_vecs = self.test_input[idx]
        # encode - apply distortion - decode
        with torch.no_grad():
            codes = self.model.encode_mu(base_vecs.to(device), update_prior=True, integrate=True)
            ranges = self.model.get_prior_range() # (min, max) for each dimension
        widths = [x_scale*(M-m)/2 for (m,M) in ranges]
        distortion_levels = [np.linspace(-w,w,steps) for w in widths]
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
        fig, ax = plt.subplots(nrows, ncols, sharey="all", figsize=figsize)
        dim=0
        for row in range(nrows):
            for col in range(ncols):
                if dim==D: break
                Nlosses = ys[:,dim,:] # N x steps
                ax[row,col].set_title(f"Latent dimension {dim}")
                # x = distortion level ; y = loss x step
                axi = sns.lineplot(np.tile(distortion_levels[dim], N), Nlosses.reshape(-1,),
                                   ax=ax[row,col], marker=".", markersize=markersize)
                ax[row,col].axvline(0, color='r', linestyle="--")
                axi.set(ylabel='L1 distance on pixels', xlabel='Latent space distortion')
                axi.tick_params(axis="x", labelsize=font_scale)
                axi.tick_params(axis="y", labelsize=font_scale)
                if ylim is not None: axi.set(ylim=(0, ylim))
                if xlim is not None: axi.set(xlim=(-xlim, xlim))
                dim+=1

        return fig

    def do_latent_distortion_multi_dim(self, latent_vector, distortion_levels, original_sample, device="cpu"):
        """ For each dimension in the original latent vector applies a sliding level of distortion
        Returns tensor of size (num_dimensions) x num_steps - where num_steps is the number of distortion levels
        applied to each dimension"""
        losses = []
        num_values = distortion_levels[0].shape[0]

        for i, dist in enumerate(distortion_levels):
            # Creates num_values copy of the latent_vector along the first axis.
            _vectors = np.tile(latent_vector.cpu().numpy(), [num_values, 1])
            # Intervenes in the latent space.
            _vectors[:, i] = dist + _vectors[:,i]
            # Generate the batch of images and computes the loss as MSE distance
            _vectors = torch.tensor(_vectors, dtype=torch.float)
            loss = self._compute_output_distortion(_vectors, original_sample, device=device)
            losses.append(loss)
        return torch.vstack(losses) # D x n_steps

    @staticmethod
    def plot_traversal_responses(dim:int, latents:Tensor, responses:Tensor, prior_samples:Tensor, **kwargs):
        """For the selected latent dimension plots the recorded errors on the responses across the traversal
        @latents: Tensor of shape (steps, N, D)
        @responses:   //        //
        kwargs accepted keys:
        - figure parameters
        - population: bool - whether to plot the whole population trend or the individual samples (default False)
        - normalise: bool - whether to normalise the codes by dividing each dimension by its standard deviation
        - relative: bool - whether to return the relative position in the traversal (i.e. distance from starting
            point) instead of the absolute one
        Returns a grid plot with D subplots.
        """
        print("")
        print(f"Plotting traversal responses for dimension {dim}")
        S, N, D = latents.shape
        population = kwargs.get('population',False)
        normalise = kwargs.get('normalise',False)
        relative = kwargs.get('relative', False)
        figsize = kwargs.get("figsize",(20,10))
        markersize = kwargs.get("markersize",10)
        font_scale = kwargs.get("font_scale",11)
        ylim = kwargs.get("ylim")
        xlim = kwargs.get("xlim")
        nrows = kwargs.get("nrows", D//3); ncols = D//nrows +1 if D%nrows!=0 else D//nrows

        i = 0
        fig, ax = plt.subplots(nrows, ncols, sharey=False, figsize=figsize)
        for row in range(nrows):
            for col in range(ncols):
                if i==D: break
                ax[row,col].set_title(f"Responses on latent dimension {i}")
                if relative:
                    x = latents[:,:,dim] - prior_samples[:,dim].view(1, -1)
                    ticks = latents[:,0,dim] - prior_samples[:,dim].mean()
                else:
                    x = latents[:,:,dim] # absolute position
                    ticks = latents[:,0,dim]
                x = x.view(-1,).cpu().numpy() # steps x N
                y = (latents[:,:,i] - responses[:,:,i]).view(-1,).cpu().numpy() # steps x N
                if normalise: y /= torch.std(responses[0,:,i]).cpu().numpy()
                hue = None if population else [i for _ in range(S) for i in range(N)]
                axi = sns.lineplot(x, y, ax=ax[row,col], marker=".", markersize=markersize, hue=hue)
                axi.axhline(0., color='red')
                axi.set(ylabel=f'Error registered on dimension {i}',
                        xlabel=f'Traversal on dimension {dim}')
                axi.set_xticklabels([f'{i:.2f}' for i in ticks])
                axi.tick_params(axis="x", labelsize=font_scale)
                axi.tick_params(axis="y", labelsize=font_scale)
                if ylim is not None: axi.set(ylim=(-ylim, ylim))
                if xlim is not None: axi.set(xlim=(-xlim, xlim))
                i+=1
        return fig

    @staticmethod
    def plot_vector_field(vectors:torch.Tensor, X:torch.Tensor, Y:torch.Tensor, **kwargs):
        """ Takes as input a set of 2D vectors (shape grid_size**2 x 2) and the corresponding
        coordinates X,Y organised in a grid (shape grid_size x grid_size) to plot them in a streamplot.
        Accepted kwargs:
        - figsize
        -i
        -j
        -type \in [stream, quiver, contour, 3D]
        -surface: whether to plot the response magnitudes or the latent response surface (exp(-response))
        """

        print("")
        print(f"Plotting vector field")
        figsize = kwargs.get("figsize",(9, 7))
        title = kwargs.get("title","Vector field")
        i = kwargs.get("i","First dim")
        j = kwargs.get("j", "Second dim")
        type = kwargs.get('type','stream')
        surface = kwargs.get('surface',False)

        grid_size = X.shape[0]
        assert vectors.shape[0] == grid_size**2, "Grid and vectors sizes not matching"
        assert vectors.shape[1] == 2, "Only 2D vector fields are supported"
        magnitudes = torch.linalg.norm(vectors, ord=2, dim=1)
        if surface: magnitudes = torch.exp(-1*magnitudes)
        U = vectors[:,0].view(grid_size,grid_size).cpu().numpy()
        V = vectors[:,1].view(grid_size,grid_size).cpu().numpy()

        fig, ax = plt.subplots(figsize=figsize)
        if type=='stream':
            strm = ax.streamplot(Y.cpu().numpy(), X.cpu().numpy(), V, U, density=2,
                                 color=magnitudes.view(grid_size,grid_size).cpu().numpy(),
                                 arrowstyle='->', arrowsize=1.5)
            fig.colorbar(strm.lines)
        elif type=="quiver":
            strm = ax.quiver(Y.cpu().numpy(), X.cpu().numpy(), V, U,
                             magnitudes.view(grid_size,grid_size).cpu().numpy())
        elif type=="contour":
            contours = plt.contourf(Y.cpu().numpy(), X.cpu().numpy(),
                                    magnitudes.view(grid_size,grid_size).cpu().numpy(),
                                    kwargs.get("contour_n",200))
            plt.colorbar()

        elif type=="3D":
            ax = fig.gca(projection='3d')
            surf = ax.plot_surface(Y.cpu().numpy(), X.cpu().numpy(),
                                   magnitudes.view(grid_size,grid_size).cpu().numpy(),
                                   rstride=1, cstride=1, linewidth=0, antialiased=False)
            fig.colorbar(surf, shrink=0.5, aspect=5)

        ax.set_title(title)
        ax.set_ylabel(f'Latent dim {i}')
        ax.set_xlabel(f'Latent dim {j}')

        return fig


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

    @staticmethod
    def plot_heatmap(matrix, **kwargs):
        """kwargs accepted keys:
        - figsize
        - thershold
        - title"""

        vmin= kwargs.get('vmin')
        vmax=kwargs.get('vmax')
        figsize = kwargs.get("figsize",(20,30))
        threshold = kwargs.get("threshold",0.01)
        title = kwargs.get("title")
        M_extreme = (matrix.abs() >= threshold)*matrix

        fig = plt.figure(figsize=figsize)
        sns.set_context('paper', font_scale=1.5)
        ax = sns.heatmap(M_extreme, linecolor='white', linewidth=2,
                         cmap="Greens", annot=True,  fmt=".2f",
                         vmin=vmin, vmax=vmax) #Values to anchor the colormap, otherwise they are inferred from the data and other keyword arguments.
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        ax.set_title(title)
        return fig

