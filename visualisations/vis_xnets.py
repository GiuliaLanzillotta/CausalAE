"""Visualisation toolkit for Xnets"""
import torch

from models import Xnet
from visualisations import utils


def compute_joint_ij(i:int, j:int, model:Xnet, device:str, **kwargs):
    """Computes joint marginal distribution over xi and xj, averaging over all noises and parent values
    Important note: at the moment the model only works for one dimensional models.
    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - marginal_samples: number of samples used to approximate the posterior over j
        - all arguments accepted in 'sample_noise_from_prior'
    Returns
    """
    print(f"Computing joint marginal between X{i} and X{j}")
    marginal_samples = kwargs.get('marginal_samples',100) # M

    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    #sample again to obtain j marginal
    _marginalj = model.sample_noise_from_prior(device=device, num_samples=marginal_samples,
                                               prior_mode="posterior").detach()[:, j]
    # traverse the latent dimension
    traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=1,
                                                      unit=j, values=_marginalj,
                                                      device=device, relative=False) # shape M x N x D
    # obtain causal variables
    all_X = model.get_causal_variables(traversals.view(-1,model.latent_size))
    Xij = torch.hstack([all_X[:,j].view(-1,1), all_X[:,i].view(-1,1)]) # shape (MxN) x 2
    hue = [i for _ in range(marginal_samples) for i in range(prior_samples.shape[0])]
    return Xij, hue


