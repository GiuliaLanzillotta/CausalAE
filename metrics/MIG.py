# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mutual Information Gap from the beta-TC-VAE paper.

Based on "Isolating Sources of Disentanglement in Variational Autoencoders"
(https://arxiv.org/pdf/1802.04942.pdf).
"""
from absl import logging
import numpy as np
from torch.utils.data import DataLoader
from . import utils

def compute_mig(dataloader:DataLoader,
                representation_function,
                num_train,
                batch_size=16,
                discretization_bins=20):
    """Computes the mutual information gap (MIG).

    Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

    Returns:
    Dict with average mutual information gap.
    """
    logging.info("Generating training set.")
    mus_train, ys_train = utils.generate_batch_factor_code(dataloader, representation_function, num_train, batch_size)
    assert mus_train.shape[1] == num_train
    # mus shape = (num_features, num_train)
    # ys shape = (num_factors, num_train)

    # Delete all factors that have only one class
    indices = np.argwhere(np.max(ys_train, axis=1) > 0).flatten() # select only rows that have entries above 0
    ys_train = ys_train[indices, :]

    # Compute MIG
    scores, extras = _compute_mig(mus_train, ys_train, bins=discretization_bins)

    # Return dictionary containing:
    #   - 'discrete_mig'
    #   - 'extras': a dict containing:
    #       - the same keys as the upper level (1 key)
    #       - 'mutual_information_matrix'

    results_dict = scores
    results_dict['extras'] = extras
    return results_dict


def _compute_mig(mus_train, ys_train, bins=20):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    # discretising the representations in order to compute mutual information
    # each row (=feature) is 'digitized' based on its range
    #TODO: what if we need to discretise the factors as well?
    discretized_mus = utils.make_discretizer(mus_train, num_bins=bins, discretizer_fn=utils._histogram_discretize)
    m = utils.discrete_mutual_info(discretized_mus, ys_train)
    assert m.shape[0] == mus_train.shape[0]
    assert m.shape[1] == ys_train.shape[0]
    # m is [num_latents, num_factors]
    entropy = utils.discrete_entropy(ys_train)
    sorted_m = np.sort(m, axis=0)[::-1]
    # for each factor get the distance in mutual information between
    # highest and second highest feature
    # normalise (btw 0 and 1) by dividing by the factor entropy
    # take the mean
    scores["discrete_mig"] = np.mean(
        np.divide(sorted_m[0, :] - sorted_m[1, :], entropy[:]))
    extras = {'mutual_information_matrix': m}
    extras.update(scores)
    return scores, extras
