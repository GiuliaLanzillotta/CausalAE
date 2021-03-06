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
"""Interventional Robustness Score.

Based on the paper https://arxiv.org/abs/1811.00007.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from absl import logging
from . import utils
import numpy as np
from torch.utils.data import DataLoader

def compute_irs(dataloader:DataLoader,
                representation_function,
                device:str,
                diff_quantile=0.99,
                num_train=100,
                batch_size=16,
                discretization_bins=20):
    """Computes the Interventional Robustness Score.

  Args:
    dataloader: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with IRS and number of active dimensions.
  """
    logging.info("Generating training set.")
    mus, ys = utils.generate_batch_factor_code(dataloader,
                                               representation_function,
                                               num_train,
                                               batch_size,
                                               device)
    assert mus.shape[1] == num_train
    # mus shape = (num_features, num_train)
    # ys shape = (num_factors, num_train)

    # discretize the factors so that we restrict ourselves to the categories we see in the training
    ys_discrete = utils.make_discretizer(ys, num_bins=discretization_bins,
                                         discretizer_fn=utils._histogram_discretize)
    active_mus, _ = utils.drop_constant_dims(mus)

    if not active_mus.any(): irs_score = 0.0
    else:
        # note: transposing dimensions (now again one data point per row)
        irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T, diff_quantile)["avg_score"]

    score_dict = {}
    score_dict["IRS"] = irs_score
    score_dict["num_active_dims"] = np.sum(active_mus)
    del mus, ys
    return score_dict


def scalable_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
    """Computes IRS scores of a dataset.

  Assumes no noise in X and crossed generative factors (i.e. one sample per
  combination of gen_factors). Assumes each g_i is an equally probable
  realization of g_i and all g_i are independent.

  Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

  Returns:
    Dictionary with IRS scores.
  """
    num_gen = gen_factors.shape[1]
    num_lat = latents.shape[1]

    # Compute normalizer.
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0) # max deviation from the mean for each feature
    cum_deviations = np.zeros([num_lat, num_gen])
    for i in range(num_gen): #for each factor
        unique_factors = np.unique(gen_factors[:, i], axis=0) # get the categories
        assert unique_factors.ndim == 1
        num_distinct_factors = unique_factors.shape[0] # number of categories
        for k in range(num_distinct_factors):
            # Compute E[Z | g_i].
            match = gen_factors[:, i] == unique_factors[k] # selecting all the samples with G_I = g_I
            e_loc = np.mean(latents[match, :], axis=0) # expected value of all latent features under intervention

            # Difference of each value within that group of constant g_i to its mean.
            # --- to be used to compute PIDA(L|empty, g_I)
            diffs = np.abs(latents[match, :] - e_loc)
            max_diffs = np.percentile(diffs, q=diff_quantile * 100, axis=0) #why not using a max?
            cum_deviations[:, i] += max_diffs
        cum_deviations[:, i] /= num_distinct_factors # expectation over intervention values
    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations # latent x factors
    disentanglement_scores = irs_matrix.max(axis=1) # for each latent
    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)

    parents = irs_matrix.argmax(axis=1)
    score_dict = {}
    score_dict["disentanglement_scores"] = disentanglement_scores
    score_dict["avg_score"] = avg_score
    score_dict["parents"] = parents
    score_dict["IRS_matrix"] = irs_matrix
    score_dict["max_deviations"] = max_deviations
    return score_dict


def full_disentanglement_score(gen_factors, latents, diff_quantile=0.99):
    #TODO: no independence assumption
    pass