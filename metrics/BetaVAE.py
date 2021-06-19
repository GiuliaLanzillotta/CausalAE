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
"""Implementation of the disentanglement metric from the BetaVAE paper.

Based on "beta-VAE: Learning Basic Visual Concepts with a Constrained
Variational Framework" (https://openreview.net/forum?id=Sy2fzU9gl).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from absl import logging
import numpy as np
from six.moves import range
from sklearn import linear_model
from datasets import DisentanglementDataset

def compute_beta_vae_sklearn(dataset:DisentanglementDataset,
                             representation_function,
                             batch_size,
                             num_train= 10000,
                             num_eval= 5000,
                             random_state=11):
    """Computes the BetaVAE disentanglement metric using scikit-learn.

  Args:
    dataset: DisentanglementDataset to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_train: Number of points used for training.
    num_eval: Number of points used for evaluation.
    random_state: For reproducibility.

  Returns:
    Dictionary with scores:
      train_accuracy: Accuracy on training set.
      eval_accuracy: Accuracy on evaluation set.
  """
    logging.info("Generating training set.")
    train_points, train_labels = _generate_training_batch(dataset, representation_function, batch_size, num_train)

    logging.info("Training sklearn model.")
    model = linear_model.LogisticRegression(random_state=random_state)
    model.fit(train_points, train_labels)

    logging.info("Evaluate training set accuracy.")
    train_accuracy = model.score(train_points, train_labels) #this will get lost?
    train_accuracy = np.mean(model.predict(train_points) == train_labels)
    logging.info("Training set accuracy: %.2g", train_accuracy)

    logging.info("Generating evaluation set.")
    eval_points, eval_labels = _generate_training_batch(dataset, representation_function, batch_size, num_eval)

    logging.info("Evaluate evaluation set accuracy.")
    eval_accuracy = model.score(eval_points, eval_labels)
    logging.info("Evaluation set accuracy: %.2g", eval_accuracy)
    scores_dict = {}
    scores_dict["train_accuracy"] = train_accuracy
    scores_dict["eval_accuracy"] = eval_accuracy
    del train_points, train_labels
    return scores_dict


def _generate_training_batch(dataset:DisentanglementDataset,
                             representation_function,
                             batch_size,
                             num_points):
    """Sample a set of training samples based on a batch of ground-truth data.
    Each sample is the average difference between pairs of feature vectors.

  Args:
    dataset: DisentanglementDataset to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    batch_size: Number of points to be used to compute the training_sample.
    num_points: Number of points to be sampled for training set.

  Returns:
    points: (num_points, dim_representation)-sized numpy array with training set
      features.
    labels: (num_points)-sized numpy array with training set labels.
  """
    points = None  # Dimensionality depends on the representation function.
    labels = np.zeros(num_points, dtype=np.int64) # labels are factor indices (this is what has to be guessed)
    for i in range(num_points):
        labels[i], feature_vector = _generate_training_sample(dataset, representation_function, batch_size)
        if points is None:
            points = np.zeros((num_points, feature_vector.shape[0]))
        points[i, :] = feature_vector
    return points, labels


def _generate_training_sample(dataset:DisentanglementDataset,
                              representation_function,
                              batch_size):
    """Sample a single training sample based on a mini-batch of ground-truth data.
    A training sample consists of the average difference between pairs of feature vectors
    obtained varying all entries but one.

  Args:
    dataset: GroundTruthData to be sampled from.
    representation_function: Function that takes observation as input and
      outputs a representation.
    batch_size: Number of points to be used to compute the training_sample

  Returns:
    index: Index of coordinate to be used.
    feature_vector: Feature vector of training sample.
  """
    # Select random coordinate to keep fixed.
    index, observations1, observations2 = dataset.sample_pairs_observations(batch_size)
    # convert to torch Tensor
    # Compute representations based on the observations.
    with torch.no_grad():
        representation1 = representation_function(observations1)
        representation2 = representation_function(observations2)
    # Compute the feature vector based on differences in representation.
    feature_vector = np.mean(np.abs(representation1 - representation2).cpu().numpy(), axis=0)
    return index, feature_vector
