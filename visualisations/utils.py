"""Utils of visualisation module"""
from typing import List

import torch
import numpy as np

from models import GenerativeAE

def get_traversals_steps(steps:int, ranges:List, relative=False):
    """Computes the values of the steps for traversal along the selected dimensions in the given ranges.
    @ranges: list of pairs (min, max) identifying the range to cover in each dimension.
    @relative:bool - whether to return the change to apply to the vector or directly the value to be adopted

    Note: returns a torch Tensor"""
    if not relative:  values = torch.vstack([torch.linspace(m,M, steps) for (m,M) in ranges])
    else:
        #move half of the total range to the right and hald of the total range to the left
        values = torch.vstack([torch.linspace(-abs(M-m)/2, abs(M-m)/2, steps) for (m,M) in ranges]) # D x N
    return values


def do_latent_traversals_multi_dim(model, latent_vector, dimensions, values, device=None):
    """ Creates a tensor where each element is obtained by passing a
        modified version of latent_vector to the generator. For each
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
            images = model.decode(torch.tensor(latent_traversal_vectors, dtype=torch.float).to(device), activate=True)
            # images has shape stepsx(image shape)
        traversals.append(images)
    return torch.cat(traversals, dim=0)

def do_latent_traversals_multi_vec(latent_vectors:torch.Tensor, unit_dim:int, unit:int, values:torch.Tensor, device:str, relative=False):
    """
    Takes a pool of latent vectors and applies multiple hard interventions to the unit 'unit'
    to traverse the latent space.
    The values to be used to intervene on the vectors are given by the 'values' parameter.
    - @latent_vectors: Tensor w/ shape (num samples, num dimensions)
    - @values: shape (num interventions, unit dimension)
    - @relative:bool - whether the given traversal values are to be considered relative to the initial position or absolute
    #TODO: include multi-dimensional units traversals

    Returns a Tensor of size (steps x N x D)
    """
    N, D = latent_vectors.shape
    steps = values.shape[0]
    values_expanded = values.view(steps,1).repeat(1, N).to(device)
    traversals_latents = torch.tile(latent_vectors, (steps, 1, 1)).to(device) # final shape = steps x N x D
    if not relative: traversals_latents[:,:,unit] = values_expanded
    else: traversals_latents[:,:,unit] += values_expanded
    return traversals_latents





