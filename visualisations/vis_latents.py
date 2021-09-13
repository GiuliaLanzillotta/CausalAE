"""Utilities for visualisation of latent space"""
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from models import GenerativeAE


def hybridiseN(model:GenerativeAE, device:str, base:Tensor=None, other:Tensor=None,  **kwargs):
    """Performs hybridisation at the noise variable level.
    A randomly picked starting example is hybridised on each dimension with another sample, randomly chosen as well.
    Returns:
        a torch.Tensor of size (B, output size), with B = 3 * D - D being the latent size of the model
        """
    single_sample = kwargs.get('single_sample',True)
    unit_dim = model.unit_dim
    complete_set = [] # here we store all the X samples to plot
    if base is None: base = model.sample_noise_from_prior(1, device=device, **kwargs).detach()
    if single_sample and other is None:
        other =  model.sample_noise_from_prior(1, device=device, **kwargs).detach()
    # new hybrid on each latent diension
    for d in range(model.latent_size):
        if not single_sample and other is None:
            other =  model.sample_noise_from_prior(1, device=device, **kwargs).detach()
        value = other[0,d*unit_dim:(d+1)*unit_dim].view(1,-1)
        intervened_sample = base.clone()
        intervened_sample[0,d*unit_dim:(d+1)*unit_dim] = value
        # we append all the vectors used to produce the new sample
        complete_set.append(base)
        complete_set.append(other)
        complete_set.append(intervened_sample)

    # this is a  3*d long tensor
    complete_set = torch.cat(complete_set, dim=0)
    with torch.no_grad(): recons = model.decode(complete_set.to(device), activate=True).detach()
    return recons, base, other

def get_posterior(model:GenerativeAE, batch_iter, device:str, **kwargs):
    """Computes aggregate posterior over the latent space.
    Returns: a tensor of shape (B x num batches) x D"""
    num_batches = kwargs.get("num_batches",10)
    _all_codes = []
    for b in range(num_batches):
        batch, _ = next(batch_iter)
        with torch.no_grad():
            codes = model.encode_mu(batch.to(device), update_prior=True, integrate=True).detach()
            _all_codes.append(codes)
    _all_codes = torch.vstack(_all_codes) # (B x num batches) x D
    return _all_codes


