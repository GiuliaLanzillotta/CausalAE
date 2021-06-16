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
"""Tests for modularity_explicitness.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
from . import utils, ModExp
import numpy as np
import torch
from torch.utils.data import DataLoader


def _identity_discretizer(target, num_bins):
    del num_bins
    return target


class ModularityTest(unittest.TestCase):

    def test_diagonal(self):
        importance_matrix = np.diag(5. * np.ones(5))
        result = ModExp.modularity(importance_matrix)
        np.testing.assert_allclose(result, 1.0)

    def test_diagonal_empty_codes(self):
        importance_matrix = np.array([[ 1., 0.,],
                                      [0., 1.],
                                      [0., 0.]])
        result = ModExp.modularity(importance_matrix)
        np.testing.assert_allclose(result, 2. / 3.)

    def test_zero(self):
        importance_matrix = np.zeros(shape=[10, 10], dtype=np.float64)
        result = ModExp.modularity(importance_matrix)
        np.testing.assert_allclose(result, .0)

    def test_redundant_codes(self):
        importance_matrix = np.diag(5. * np.ones(5))
        importance_matrix = np.vstack([importance_matrix, importance_matrix])
        result = ModExp.modularity(importance_matrix)
        np.testing.assert_allclose(result, 1.)

    def test_missed_factors(self):
        importance_matrix = np.diag(5. * np.ones(5))
        result = ModExp.modularity(importance_matrix[:2, :])
        np.testing.assert_allclose(result, 1.0)

    def test_one_code_two_factors(self):
        importance_matrix = np.diag(5. * np.ones(5))
        importance_matrix = np.hstack([importance_matrix, importance_matrix])
        result = ModExp.modularity(importance_matrix)
        np.testing.assert_allclose(result, 1. - 1. / 9)


class ModularityExplicitnessTest(unittest.TestCase):

    def test_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: x
        scores = ModExp.compute_modularity_explicitness(dataloader, representation_function,
                                                        3000, 3000, 100, 10)
        self.assertTrue(0.9 <= scores["modularity_score"] <= 1.0)

    def test_bad_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        # The representation which randomly permutes the factors, should have equal
        # non-zero MI which should give a low modularity score.
        def representation_function(x):
            code = np.array(x, dtype=np.float64)
            for i in range(code.shape[0]):
                code[i, :] = np.random.permutation(code[i, :])
            return torch.Tensor(code)

        scores =  ModExp.compute_modularity_explicitness(dataloader, representation_function,
                                                         20000, 20000, 100, 10)
        self.assertTrue(0.0 <= scores["modularity_score"] <= 0.2)

    def test_duplicated_latent_space(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        def representation_function(x):
            return torch.hstack([x, x])

        scores = ModExp.compute_modularity_explicitness(dataloader, representation_function,
                                                        3000, 3000, 100, 10)
        self.assertTrue(0.9 <= scores["modularity_score"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
