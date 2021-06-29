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
"""Modularity and explicitness metrics from the F-statistic paper.

Based on "Learning Deep Disentangled Embeddings With the F-Statistic Loss"
(https://arxiv.org/pdf/1802.05312.pdf).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from . import utils
import numpy as np
from six.moves import range
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader


def compute_modularity_explicitness(dataloader:DataLoader,
                                    representation_function,
                                    device:str,
                                    num_train=10000,
                                    num_test=5000,
                                    batch_size=16,
                                    discretization_bins=20):
    """Computes the modularity metric according to Sec 3.

  Args:
    dataloader: data to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.
    discretization_bins: number of bins to be used in discretization of mu

  Returns:
    Dictionary with average modularity score and average explicitness
      (train and test).
  """
    scores = {}
    mus_train, ys_train = utils.generate_batch_factor_code(dataloader, representation_function, num_train, batch_size, device)
    # mus shape = (num_features, num_train)
    # ys shape = (num_factors, num_train)
    mus_test, ys_test = utils.generate_batch_factor_code(dataloader, representation_function, num_test, batch_size, device)

    # Delete all factors that have only one class
    all_labels = np.concatenate([ys_train, ys_test], axis=1)
    indices = np.argwhere(np.max(all_labels, axis=1) > 0).flatten()
    ys_train = ys_train[indices, :]
    ys_test = ys_test[indices, :]

    discretized_mus = utils.make_discretizer(mus_train, num_bins=discretization_bins,
                                             discretizer_fn=utils._histogram_discretize)
    mutual_information = utils.discrete_mutual_info(discretized_mus, ys_train)
    # Mutual information should have shape [num_codes, num_factors].
    assert mutual_information.shape[0] == mus_train.shape[0]
    assert mutual_information.shape[1] == ys_train.shape[0]
    scores["modularity_score"] = modularity(mutual_information)
    explicitness_score_train = np.zeros([ys_train.shape[0], 1])
    explicitness_score_test = np.zeros([ys_test.shape[0], 1])
    mus_train_norm, mean_mus, stddev_mus = utils.normalize_data(mus_train)
    mus_test_norm, _, _ = utils.normalize_data(mus_test, mean_mus, stddev_mus)
    for i in range(ys_train.shape[0]): # for each factor
        explicitness_score_train[i], explicitness_score_test[i] = \
            explicitness_per_factor(mus_train_norm, ys_train[i, :],
                                    mus_test_norm, ys_test[i, :])
    scores["explicitness_score_train"] = np.mean(explicitness_score_train)
    scores["explicitness_score_test"] = np.mean(explicitness_score_test)
    del mus_train, ys_train, mus_test, ys_test
    return scores


def explicitness_per_factor(mus_train, y_train, mus_test, y_test):
    """Compute explicitness score for a factor as ROC-AUC of a classifier.
    Basically measuring how well we can predict the given factor.

  Args:
    mus_train: Representation for training, (num_codes, num_points)-np array.
    y_train: Ground truth factors for training, (num_factors, num_points)-np
      array.
    mus_test: Representation for testing, (num_codes, num_points)-np array.
    y_test: Ground truth factors for testing, (num_factors, num_points)-np
      array.

  Returns:
    roc_train: ROC-AUC score of the classifier on training data.
    roc_test: ROC-AUC score of the classifier on testing data.
  """
    x_train = np.transpose(mus_train) # num_points x features
    x_test = np.transpose(mus_test) # num_points x features
    clf = LogisticRegression().fit(x_train, y_train)
    y_pred_train = clf.predict_proba(x_train)
    y_pred_test = clf.predict_proba(x_test)
    mlb = MultiLabelBinarizer()
    roc_train = roc_auc_score(mlb.fit_transform(np.expand_dims(y_train, 1)), y_pred_train)
    roc_test = roc_auc_score(mlb.transform(np.expand_dims(y_test, 1)), y_pred_test)
    return roc_train, roc_test


def modularity(mutual_information):
    """Computes the modularity from mutual information.
    Basically measuring how much each feature is specific to one factor."""
    # Mutual information has shape [num_codes, num_factors].
    squared_mi = np.square(mutual_information)
    max_squared_mi = np.max(squared_mi, axis=1) # for each feature get max factor
    numerator = np.sum(squared_mi, axis=1) - max_squared_mi # the bigger this is the more factors each feature shares information with
    denominator = max_squared_mi * (squared_mi.shape[1] - 1.) # normalising (maximum = MAX*num_factors)
    delta = numerator / denominator
    modularity_score = 1. - delta
    index = (max_squared_mi == 0.)
    modularity_score[index] = 0.
    return np.mean(modularity_score)
