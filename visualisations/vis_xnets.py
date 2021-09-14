"""Visualisation toolkit for Xnets"""
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

import models.LVAE
from models import Xnet
from models import utils as mutils
from visualisations import utils
from sklearn.decomposition import PCA


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

def hybridiseX(model:Xnet, device:str, base:Tensor=None, other:Tensor=None, **kwargs):
    """Performs hybridisation at the causal variable level.
    A randomly picked starting example is hybridised on each dimension with another sample, randomly chosen as well.
    Returns:
        a torch.Tensor of size (B, output size), with B = 3 * D - D being the latent size of the model
        """
    single_sample = kwargs.get('single_sample',True)

    complete_set = [] # here we store all the X samples to plot
    if base is None: base = model.sample_noise_from_prior(1, device=device, **kwargs)
    baseX = model.get_causal_variables(base, **kwargs).detach()
    if single_sample:
        if other is None: other =  model.sample_noise_from_prior(1, device=device, **kwargs)
        otherX = model.get_causal_variables(other, **kwargs).detach()
    # new hybrid on each latent diension
    for d in range(model.latent_size):
        if not single_sample:
            if other is None: other =  model.sample_noise_from_prior(1, device=device, **kwargs)
            otherX = model.get_causal_variables(other, **kwargs).detach()
        value = otherX[0,d*model.xunit_dim:(d+1)*model.xunit_dim].view(1,-1)
        intervened_sample = model.intervene_on_X(d, base, value).detach()
        # we append all the vectors used to produce the new sample
        complete_set.append(baseX)
        complete_set.append(otherX)
        complete_set.append(intervened_sample)

    # this is a  3*d long tensor
    complete_set = torch.cat(complete_set, dim=0)
    with torch.no_grad(): recons = model.decode_from_X(complete_set.to(device), activate=True).detach()
    return recons, base, other

def multidimUnitMarginal(model:Xnet, unit:int, device:str, **kwargs):
    """Computes 2D projection of multidimensional unit marginal aggregate posterior distribution"""
    # 1. obtain the samples
    samples =  model.sample_noise_from_prior(device=device, **kwargs).detach()
    samples_ux = model.get_causal_variables(samples, **kwargs)[:,model.xunit_dim*unit:model.xunit_dim*(unit+1)].detach()
    # 2. project into 2D space
    if model.xunit_dim ==2: return samples_ux.cpu().numpy()
    pca = PCA(n_components=2)
    samples_ux2D = pca.fit_transform(samples_ux.cpu().numpy())
    return samples_ux2D

def get_posterior(model:Xnet, batch_iter, device:str, **kwargs):
    """Computes aggregate posterior over the causal variables  space.
    Returns: a tensor of shape (B x num batches) x D"""
    num_batches = kwargs.get("num_batches",10)
    _all_X = []
    for b in range(num_batches):
        batch, _ = next(batch_iter)
        with torch.no_grad():
            noises = model.encode_mu(batch.to(device), update_prior=True, integrate=True).detach()
            X = model.get_causal_variables(noises, **kwargs)
            _all_X.append(X)
    _all_X = torch.vstack(_all_X) # (B x num batches) x D
    return _all_X

def compute_N2X(dim:int,model:Xnet, device:str, **kwargs):
    """Computes noise to X joint
    - only works for 1 dimensional units models"""
    print(f"Computing joint between N{dim} and X{dim}")
    marginal_samples = kwargs.get('marginal_samples',100) # M
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    #sample again to obtain noise dim marginal
    _marginalN = model.sample_noise_from_prior(device=device, num_samples=marginal_samples,
                                               prior_mode="posterior").detach()[:, dim]
    # traverse the latent dimension
    traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=1,
                                                      unit=dim, values=_marginalN,
                                                      device=device, relative=False).view(-1,model.latent_size) # shape M x N x D
    # obtain causal variables
    all_X = model.get_causal_variables(traversals)
    NX = torch.hstack([traversals[:,dim].view(-1,1), all_X[:,dim].view(-1,1)]) # shape (MxN) x 2
    hue = [i for _ in range(marginal_samples) for i in range(prior_samples.shape[0])]
    return NX, hue

def compute_renyis_entropy_X(model:Xnet,  batch_iter:Iterable, device:str, **kwargs):
    """ Computes renyis entropy over each causal dimension for alpha = 2
    source: http://www.cnel.ufl.edu/courses/EEL6814/renyis_entropy.pdf
    """
    print(f"Computing entropy of the causal variables")

    # collect samples from aggregate posterior on X
    num_batches = kwargs.get("num_batches",10)
    _all_X = []
    for b in range(num_batches):
        batch, _ = next(batch_iter)
        with torch.no_grad():
            noises = model.encode_mu(batch.to(device), update_prior=True, integrate=True).detach()
            X = model.get_causal_variables(noises, **kwargs).detach()
            _all_X.append(X)
    # concatenating and reshaping
    _all_X = torch.vstack(_all_X).view(-1, model.latent_size, model.xunit_dim) # N x L x D
    N,_,D = _all_X.shape
    entropies = torch.zeros(model.latent_size, device=device)
    for d in range(model.latent_size):
        # compute optimal sigma
        sigma = torch.std(_all_X[:,d,:]).to(device)
        mulC = math.pow((4/N)*(1/(2*D+1)),(1/(D+4)))
        sigma= sigma*mulC
        #obtain kernel estimates of the distances of all samples
        _all_G = mutils.Gaussian_kernel(_all_X[:,d,:], sigma)
        # estimate entropy
        entropies[d] = -1*torch.log(torch.mean(_all_G)).to(device)
    return entropies



