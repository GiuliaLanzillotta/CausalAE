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
