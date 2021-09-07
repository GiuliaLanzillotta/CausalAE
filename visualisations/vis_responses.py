""" All functions needed to compute the material to visualise the latent responses"""
import torch

from models import GenerativeAE
from . import utils


def traversal_responses(model:GenerativeAE, device, **kwargs):
    """Plots the amount of distortion recorded on each latent dimension while traversing a single dimension
    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - steps: number of steps to take in the traversal
        - all arguments accepted in 'sample_noise_from_prior'

    Returns 2 lists containing the latents and corresponding responses for each latent unit
    #TODO: add support for multidimensional units
    """
    print("Computing traversal responses ... ")
    if not kwargs.get('num_samples'):
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps',20)
    unit_dim = 1
    num_units = model.latent_size//unit_dim

    all_traversal_latents = []
    all_traversals_responses = []
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(**kwargs).to(device).detach()
    ranges = model.get_prior_range()
    # for each latent unit we start traversal
    for u in range(num_units):
        range_u = ranges[u]
        # 1. obtain traversals values
        traversals_steps = utils.get_traversals_steps(steps, [range_u]).to(device).detach() #torch Tensor
        with torch.no_grad():
            # 2. do traversals
            traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=unit_dim, unit=u,
                                                              values=traversals_steps, device=device) # shape steps x N x D
            traversals_latents = traversals.view(-1, model.latent_size) # reshaping to fit into batch
            # 3. obtain responses
            trvs_response = model.encode_mu(model.decode(traversals_latents, activate=True))
            all_traversal_latents.append(traversals_latents)
            all_traversals_responses.append(trvs_response)

    print("...done")

    return all_traversal_latents, all_traversals_responses

def hybrid_responses():
    #TODO
    pass

def response_field(i:int, j:int, model:GenerativeAE, device, **kwargs):
    """Evaluates the response field over the selected latent dimensions (i and j are the indices)
    kwargs accepted keys:
    - grid_size
    - num_samples
    - all kwargs accepted in 'sample noise from prior'
    #TODO: add support for multidimensional units

    Returns tensor of size (grid_size**2, 2) with the mean responses over the grid and the grid used
    """

    print(f"Computing response field for {i} and {j} ... ")
    grid_size = kwargs.get('grid_size', 20) #400 points by default
    num_samples = kwargs.get('num_samples',50)
    unit_dim = 1
    num_units = model.latent_size//unit_dim
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(**kwargs).to(device).detach()
    ranges = model.get_prior_range()
    with torch.no_grad():
        hybrid_grid = torch.meshgrid([torch.linspace(ranges[i][0],ranges[i][1], steps=grid_size),
                                      torch.linspace(ranges[j][0],ranges[j][1], steps=grid_size)])
        i_values = hybrid_grid[0].contiguous().view(-1, unit_dim) # M x u
        j_values = hybrid_grid[1].contiguous().view(-1, unit_dim) # M x u
        assert i_values.shape[0] == grid_size**2, "Something wrong detected with meshgrid"
        # now for each of the prior samples we want to evaluate the full grid in order to then average the results  (mean field approximation)
        all_samples = torch.tile(prior_samples, (grid_size**2,1,1)) #shape = M x N x D (M is grid_size**2)
        all_samples[:,:,i] = i_values.repeat(1,num_samples)
        all_samples[:,:,j] = j_values.repeat(1,num_samples)
        responses = model.encode_mu(model.decode(all_samples.view(-1,num_units), activate=True))
        response_field = torch.hstack([responses[:,i], responses[:,j]]).view(grid_size**2, num_samples, 2).mean(dim=1) # M x 2

    return response_field, hybrid_grid


