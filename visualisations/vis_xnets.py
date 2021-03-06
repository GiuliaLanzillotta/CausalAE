"""Visualisation toolkit for Xnets"""
import math
from typing import Iterable

import numpy as np
import torch
from torch import Tensor

import models.LVAE
from experiments import get_causal_block_graph
from experiments.utils import temperature_exponential_annealing
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

def double_hybridiseX(model:Xnet, device:str, prior_samples=None, **kwargs):
    """Performs hybridisation at the causal variable level.
    A randomly picked starting example is hybridised on each pair of dimensions with
     two other samples randomly chosen as well.
    Returns:
        a torch.Tensor of size (B, output size), with B = (D+1)) * (D+1) - D being the latent size of the model
        """
    xunit_dim = model.xunit_dim
    complete_set = []
    if prior_samples is None:
        #first one is base tensor, second and third are used to intervene on the base
        all_samples =  model.sample_noise_from_prior(3, device=device, **kwargs)
    else: all_samples = prior_samples
    all_samplesX = model.get_causal_variables(all_samples)
    # necessary for visualisation purposes
    # for each pair of dimensions we obtain a new intervention with the two samples
    for i in range(model.latent_size):
        complete_set.append(all_samplesX[1].view(1,-1))
        value_i = all_samplesX[1,i*xunit_dim:(i+1)*xunit_dim].view(1,1,xunit_dim)
        for j in range(model.latent_size):
            if i==j: # diagonals only accept one intervention
                intervened_sample = model.intervene_on_X([i], all_samples[0].view(1,-1), value_i).detach()
                # we append all the vectors used to produce the new sample
                complete_set.append(intervened_sample)
                continue
            value_j = all_samplesX[2,j*xunit_dim:(j+1)*xunit_dim].view(1,1,xunit_dim)
            values = torch.cat([value_i, value_j], dim=1)
            intervened_sample = model.intervene_on_X([i,j], all_samples[0].view(1,-1), values).detach()
            # we append all the vectors used to produce the new sample
            complete_set.append(intervened_sample)

    # necessary for visualisation purposes
    complete_set.append(all_samplesX[0].view(1,-1))
    for i in range(model.latent_size): complete_set.append(all_samplesX[2].view(1,-1))
    # this is a d+1 * d+1 long tensor - dxd matrix of images out
    complete_set = torch.cat(complete_set, dim=0)
    with torch.no_grad(): recons = model.decode_from_X(complete_set.to(device), activate=True).detach()
    return recons


def hybridiseX(model:Xnet, device:str, base:Tensor=None, other:Tensor=None, **kwargs):
    """Performs hybridisation at the causal variable level.
    A randomly picked starting example is hybridised on each dimension with another sample, randomly chosen as well.
    Returns:
        a torch.Tensor of size (B, output size), with B = 3 * D - D being the latent size of the model
        """
    single_sample = kwargs.get('single_sample',True)
    xunit_dim = model.xunit_dim
    complete_set = [] # here we store all the X samples to plot
    if base is None: base = model.sample_noise_from_prior(1, device=device, **kwargs)
    baseX = model.get_causal_variables(base, **kwargs).detach()
    if single_sample:
        if other is None: other =  model.sample_noise_from_prior(1, device=device, **kwargs)
        otherX = model.get_causal_variables(other, **kwargs).detach().view(1,-1, xunit_dim)
    # new hybrid on each latent diension
    for d in range(model.latent_size):
        if not single_sample:
            if other is None: other =  model.sample_noise_from_prior(1, device=device, **kwargs)
            otherX = model.get_causal_variables(other, **kwargs).detach().view(1,-1, xunit_dim)
        value = otherX[0,d*model.xunit_dim:(d+1)*model.xunit_dim].view(1,-1)
        intervened_sample = model.intervene_on_X([d], base, value).detach()
        # we append all the vectors used to produce the new sample
        complete_set.append(baseX)
        complete_set.append(otherX)
        complete_set.append(intervened_sample)

    # this is a  3*d long tensor
    complete_set = torch.cat(complete_set, dim=0)
    with torch.no_grad(): recons = model.decode_from_X(complete_set.to(device), activate=True).detach()
    return recons, base, other


def traversalsX(model:Xnet, device:str, dim:int, inputs=None, **kwargs):
    """Traverses causal latent space on the specified dimension by applying interventions."""

    print("Computing causal latent traversals...")
    if not kwargs.get('num_samples'): #note: this argument will be ignored if inputs is provided
        kwargs['num_samples'] = 50
    steps = kwargs.get('steps',20)
    ranges = model.estimate_range_dim(dim, device, **kwargs)

    with torch.no_grad():
        if inputs is None: codes = model.sample_noise_from_prior(device=device, **kwargs).detach()
        else: codes = model.encode_mu(inputs.to(device))  # sample randomly
        X = model.get_causal_variables(codes, **kwargs).detach().reshape(-1, model.latent_size, model.xunit_dim)
        traversal_vecs = []
        for sub_dim in range(model.xunit_dim):
            traversals_steps = utils.get_traversals_steps(steps, [ranges[sub_dim]], relative=False).to(device).detach()[0]
            for v in traversals_steps:
                vec = X[:,dim,:]
                vec[:,sub_dim] = v # preparing intervention values in the format required by Xnet intervention
                I_X = model.intervene_on_X([dim], codes, torch.unsqueeze(vec,1)).detach() # num_samples x L x U
                traversal_vecs.append(I_X)
        traversal_vecs = torch.vstack(traversal_vecs) # (num_samples x steps x unit_dim) x L
        recons = model.decode_from_X(traversal_vecs.to(device), activate=True).detach()
    print("...done")
    return recons

def get_free_causes(model:Xnet, device:str, **kwargs):
    """Returns list containing indices of free causal variables according
    to the Xnet structure"""
    tau = kwargs.get('tau', temperature_exponential_annealing(100000))
    A = get_causal_block_graph(model, "X", device, tau=tau).detach().cpu().numpy()
    incoming_weight = A.sum(axis=0)
    idx = list(np.where(incoming_weight==0)[0])
    return idx






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

def compute_N2X(dimN:int, dimX:int, model:Xnet, device:str, **kwargs):
    """Computes noise to X joint"""
    print(f"Computing joint between N{dimN} and X{dimX}")
    marginal_samples = kwargs.get('marginal_samples',100) # M
    # sample N vectors from prior
    prior_samples = model.sample_noise_from_prior(device=device, **kwargs).detach()
    #sample again to obtain noise dim marginal
    _marginalN = model.sample_noise_from_prior(device=device, num_samples=marginal_samples,
                                               prior_mode="posterior").detach()[:, dimN]
    # traverse the latent dimension
    traversals = utils.do_latent_traversals_multi_vec(prior_samples, unit_dim=1,
                                                      unit=dimN, values=_marginalN,
                                                      device=device, relative=False).view(-1,model.latent_size) # shape M x N x D
    # obtain causal variables
    all_X = model.get_causal_variables(traversals).view(-1, model.latent_size, model.xunit_dim) # (MxN) x Dx
    NX = torch.hstack([traversals[:,dimN].view(-1,1), all_X[:,dimX,:]]) # shape (MxN) x (1 + xunit_dim)
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



