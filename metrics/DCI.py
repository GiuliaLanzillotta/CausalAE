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
"""Implementation of Disentanglement, Completeness and Informativeness.

Based on "A Framework for the Quantitative Evaluation of Disentangled
Representations" (https://openreview.net/forum?id=By-7dz-AZ).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import logging
from . import utils
import numpy as np
import scipy
import scipy.stats
from six.moves import range
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import DataLoader



def compute_dci(dataloader:DataLoader,
                representation_function,
                num_train=10000,
                num_test=5000,
                batch_size=16):
    """Computes the DCI scores according to Sec 2.

    Args:
      dataloader: data to be sampled from.
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      batch_size: Batch size for sampling.

    Returns:
      Dictionary with average disentanglement score, completeness and
        informativeness (train and test).
    """
    logging.info("Generating training set.")
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    mus_train, ys_train = utils.generate_batch_factor_code(dataloader, representation_function, num_train, batch_size)
    assert mus_train.shape[1] == num_train
    assert ys_train.shape[1] == num_train
    mus_test, ys_test = utils.generate_batch_factor_code(dataloader, representation_function, num_test, batch_size) #TODO: maybe factorise this duplicated code

    # Delete all factors that have only one class
    all_labels = np.concatenate([ys_train, ys_test], axis=1)
    indices = np.argwhere(np.max(all_labels, axis=1) > 0).flatten()
    ys_train = ys_train[indices, :]
    ys_test = ys_test[indices, :]

    # Compute DCI metrics
    scores, extras = _compute_dci(mus_train, ys_train, mus_test, ys_test)

    # Return dictionary containing:
    #   - 'informativeness_train'
    #   - 'informativeness_test'
    #   - 'disentanglement'
    #   - 'completeness'
    #   - 'extras': a dict containing:
    #       - the same 4 keys as the upper level
    #       - 'importance_matrix'
    #       - 'informativeness_train_per_factor'
    #       - 'informativeness_test_per_factor'
    #       - 'disentanglement_loo_factor'
    #       - 'disentanglement_loo_latent'
    #       - 'disentanglement_loo_factor_latent'
    #       - 'completeness_loo_factor'
    #       - 'completeness_loo_latent'
    #       - 'completeness_loo_factor_latent'

    results_dict = scores
    results_dict['extras'] = extras
    return results_dict


def _compute_dci(mus_train, ys_train, mus_test, ys_test):
    """Computes score based on both training and testing codes and factors."""
    scores = {}
    importance_matrix, train_err, test_err = compute_importance_gbt(mus_train, ys_train,
                                                                    mus_test, ys_test)
    assert importance_matrix.shape[0] == mus_train.shape[0]
    assert importance_matrix.shape[1] == ys_train.shape[0]
    scores["informativeness_train"] = np.mean(train_err)  # avg mean-accuracy over factors
    scores["informativeness_test"] = np.mean(test_err)  # avg mean-accuracy over factors
    scores["disentanglement"] = disentanglement(importance_matrix)
    scores["completeness"] = completeness(importance_matrix)

    # Get leave-one-out disentanglement and completeness (leaving out one factor, one latent, one pair factor-latent)
    d_loo_factor, c_loo_factor = get_metrics_loo_factor(importance_matrix)
    d_loo_latent, c_loo_latent = get_metrics_loo_latent(importance_matrix)
    d_loo_factor_latent, c_loo_factor_latent = get_metrics_loo_factor_latent(importance_matrix)

    extras = {
        'importance_matrix': importance_matrix,
        'informativeness_train_per_factor': train_err,
        'informativeness_test_per_factor': test_err,
        'disentanglement_loo_factor': d_loo_factor,
        'disentanglement_loo_latent': d_loo_latent,
        'disentanglement_loo_factor_latent': d_loo_factor_latent,
        'completeness_loo_factor': c_loo_factor,
        'completeness_loo_latent': c_loo_latent,
        'completeness_loo_factor_latent': c_loo_factor_latent,
    }
    extras.update(scores)
    return scores, extras


def compute_importance_gbt(x_train, y_train, x_test, y_test):
    """Compute importance based on gradient boosted trees.
    Importance is calculated for a single decision tree by the amount that each attribute split
    point improves the performance measure, weighted by the number of observations the node
    is responsible for. The feature importances are then averaged across all of the the
    decision trees within the model.
    -- Basically measuring correlation btw dimension and residual --
    """
    num_factors = y_train.shape[0]
    num_codes = x_train.shape[0]
    importance_matrix = np.zeros(shape=[num_codes, num_factors], dtype=np.float64) # a row for each latent dim, column for each factor
    train_loss = []
    test_loss = []
    for i in range(num_factors):
        model = GradientBoostingClassifier()
        # x_train.T is (num_samples, sample_dim)
        model.fit(x_train.T, y_train[i, :]) # predicting i-th factor across all samples
        importance_matrix[:, i] = np.abs(model.feature_importances_) # why abs?
        # measuring average accuracy on train and test set
        train_loss.append(np.mean(model.predict(x_train.T) == y_train[i, :]))
        test_loss.append(np.mean(model.predict(x_test.T) == y_test[i, :]))
    train_loss = np.asarray(train_loss)  # per factor
    test_loss = np.asarray(test_loss)  # per factor
    return importance_matrix, train_loss, test_loss


def disentanglement_per_code(importance_matrix):
    """Compute disentanglement score of each code dimension as 1 - importance entropy."""
    # importance_matrix is of shape [num_codes, num_factors].
    # entropy is calculated along 0 axis (per columns) --> calculate entropy for each code
    return 1. - scipy.stats.entropy(importance_matrix.T + 1e-11,
                                    base=importance_matrix.shape[1]) # logarithmic base used = num_factors -> normalised entropy (uniform has entropy 1)


def disentanglement(importance_matrix):
    """Compute the disentanglement score of the representation.
    Disentanglement = normalised dimension->factors entropy * dimension relative importance"""
    per_code = disentanglement_per_code(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    # importance matrix was not normalised
    code_importance = importance_matrix.sum(axis=1) / importance_matrix.sum()

    return np.sum(per_code * code_importance)


def completeness_per_factor(importance_matrix):
    """Compute completeness of each factor.
    Basically calculating the (normalised) entropy of each factor distribution"""
    # importance_matrix is of shape [num_codes, num_factors].
    return 1. - scipy.stats.entropy(importance_matrix + 1e-11,
                                    base=importance_matrix.shape[0])


def completeness(importance_matrix):
    """"Compute completeness of the representation.
    Completeness = normalised factor->dimensions entropy * factor relative importance"""
    per_factor = completeness_per_factor(importance_matrix)
    if importance_matrix.sum() == 0.:
        importance_matrix = np.ones_like(importance_matrix)
    factor_importance = importance_matrix.sum(axis=0) / importance_matrix.sum()
    return np.sum(per_factor * factor_importance)


def get_metrics_loo_factor_latent(matrix):
    n_latents = matrix.shape[0]
    n_factors = matrix.shape[1]
    disentanglement_loo = np.zeros((n_latents, n_factors))
    completeness_loo = np.zeros((n_latents, n_factors))
    for r, c in itertools.product(range(n_latents), range(n_factors)):
        rows = [i for i in range(n_latents) if i != r]
        cols = [i for i in range(n_factors) if i != c]
        loo_matrix = matrix[rows]
        loo_matrix = loo_matrix[:, cols]
        disentanglement_loo[r, c] = disentanglement(loo_matrix)
        completeness_loo[r, c] = completeness(loo_matrix)
    return disentanglement_loo, completeness_loo


def get_metrics_loo_latent(matrix):
    n_latents = matrix.shape[0]
    disentanglement_loo = np.zeros((n_latents,))
    completeness_loo = np.zeros((n_latents,))
    for i in range(n_latents):
        indices = [j for j in range(n_latents) if j != i]
        loo_matrix = matrix[indices, :]
        disentanglement_loo[i] = disentanglement(loo_matrix)
        completeness_loo[i] = completeness(loo_matrix)
    return disentanglement_loo, completeness_loo


def get_metrics_loo_factor(matrix):
    n_factors = matrix.shape[1]
    disentanglement_loo = np.zeros((n_factors,))
    completeness_loo = np.zeros((n_factors,))
    for i in range(n_factors):
        indices = [j for j in range(n_factors) if j != i]
        loo_matrix = matrix[:, indices]
        disentanglement_loo[i] = disentanglement(loo_matrix)
        completeness_loo[i] = completeness(loo_matrix)
    return disentanglement_loo, completeness_loo
