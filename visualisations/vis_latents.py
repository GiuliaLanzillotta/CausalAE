"""Utilities for visualisation of latent space"""
from collections import Iterable

import torch
from torch import Tensor
import numpy as np
from . import utils

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

def double_hybridiseN(model:GenerativeAE, device:str, **kwargs):
    """Performs double hybridisation at the noise variable level.
    A randomly picked starting example is hybridised on each pair of dimensions with
     two other samples randomly chosen as well.
    Returns:
        a torch.Tensor of size (B, output size), with B = (D+1)) * (D+1) - D being the latent size of the model
        """
    complete_set = []
    #first one is base tensor, second and third are used to intervene on the base
    all_samples =  model.sample_noise_from_prior(3, device=device, **kwargs).detach()
    # necessary for visualisation purposes
    # for each pair of dimensions we obtain a new intervention with the two samples
    for i in range(model.latent_size):
        complete_set.append(all_samples[1].view(1,-1))
        value_i = all_samples[1,i]
        for j in range(model.latent_size):
            if i==j: # diagonals only accept one intervention
                intervened_sample = all_samples[0].clone().view(1,-1)
                intervened_sample[:,i] = value_i
                # we append all the vectors used to produce the new sample
                complete_set.append(intervened_sample)
                continue
            value_j = all_samples[2,j]
            intervened_sample = all_samples[0].clone().view(1,-1)
            intervened_sample[:,i] = value_i
            intervened_sample[:,j] = value_j
            # we append all the vectors used to produce the new sample
            complete_set.append(intervened_sample)

    # necessary for visualisation purposes
    complete_set.append(all_samples[0].view(1,-1))
    for i in range(model.latent_size): complete_set.append(all_samples[2].view(1,-1))
    # this is a d+1 * d+1 long tensor - dxd matrix of images out
    complete_set = torch.cat(complete_set, dim=0)
    with torch.no_grad(): recons = model.decode(complete_set.to(device), activate=True).detach()
    return recons, all_samples

def interpolate(model:GenerativeAE, batch_iter:Iterable, device:str, **kwargs):
    """Interpolates between two sample in the posterior"""
    seed = kwargs.get("random_seed",13)
    rndn_gen = np.random.RandomState(seed)
    nsteps = kwargs.get("num_steps",20)

    codes_path = [] # here all the codes  will be added
    batch, _ = next(batch_iter)
    with torch.no_grad():
        codes = model.encode_mu(batch.to(device), update_prior=True, integrate=True).detach()
    start_code = codes[rndn_gen.randint(codes.shape[0])]
    end_code = codes[rndn_gen.randint(codes.shape[0])]
    # 0.0-> start code/ 1.0-> end code
    weights = torch.linspace(0.0, 1.0, steps=nsteps, device=device)
    for w in weights:
        # linear interpolation
        codes_path.append(torch.lerp(start_code, end_code, w).view(1,-1).to(device))
    codes_path = torch.vstack(codes_path).to(device)

    # switchig to output space
    out_path = model.decode(codes_path, activate=True).detach()
    return out_path


def traversals(model:GenerativeAE, device, inputs=None, **kwargs):
    """Computes the reconstructions obtained while traversing a single dimension.
    If inputs is given the codes for the traversals are randomly picked from the latent representation
    of the inputs.
    kwargs accepted keywords:
        - num_samples: number of samples from the latent space to use in the computation
        - steps: number of steps to take in the traversal
        - all arguments accepted in 'sample_noise_from_prior'

    Returns 2 lists containing the latents and corresponding responses for each latent unit
    """
    print("Computing latent traversals ... ")
    if not kwargs.get('num_samples'):
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps',20)
    unit_dim = 1

    if inputs is None: codes = model.sample_noise_from_prior(device=device, **kwargs).detach()
    else: codes = model.encode_mu(inputs[torch.randperm(inputs.shape[0])[:kwargs['num_samples']]].to(device)) #sample randomly

    all_reconstructions = []
    # sample N vectors from prior

    ranges = model.get_prior_range()
    # for each latent unit we start traversal
    # 1. obtain traversals values
    traversals_steps = utils.get_traversals_steps(steps, ranges, relative=False).to(device).detach() #torch Tensor
    for d in range(model.latent_size):
        with torch.no_grad():
            # 2. do traversals
            traversal_vecs = utils.do_latent_traversals_multi_vec(codes, unit_dim=unit_dim,
                                                                  unit=d, values=traversals_steps[d],
                                                                  device=device, relative=False).detach() # shape steps x N x D
            traversal_vecs = traversal_vecs.view(-1, model.latent_size) # reshaping to fit into batch
            # 3. obtain responses
            recons = model.decode(traversal_vecs.to(device), activate=True).detach() #shape = (SxN) x X_dim
            all_reconstructions.append(recons)

    print("...done")

    return torch.cat(all_reconstructions, dim=0).to(device)


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


