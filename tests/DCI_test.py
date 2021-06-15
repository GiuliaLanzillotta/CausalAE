import unittest
from . import utils
from . import DCI
import numpy as np
from torch.utils.data import DataLoader
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
"""Tests for dci_test.py."""
#TODO

class DCITest(unittest.TestCase):

    def test_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: np.array(x, dtype=np.float64)
        scores = DCI.compute_dci(dataloader, representation_function, 1000, 1000, 100)
        self.assertTrue(0.9 <=scores["disentanglement"] <= 1.0)
        self.assertTrue(0.9 <= scores["completeness"] <= 1.0)

    def test_bad_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        # The representation which randomly permutes the factors, should have equal
        # non-zero importance which should give a low modularity score.
        def representation_function(x):
            code = np.array(x, dtype=np.float64)
            for i in range(code.shape[0]):
                code[i, :] = np.random.permutation(code[i, :])
            return code

        scores = DCI.compute_dci(dataloader, representation_function, 1000, 1000, 100)
        self.assertTrue(0.0 <= scores["disentanglement"] <= 0.2)
        self.assertBetween(0.0 <= scores["completeness"] <= 0.2)

    def test_duplicated_latent_space(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        def representation_function(x):
            x = np.array(x, dtype=np.float64)
            return np.hstack([x, x])

        random_state = np.random.RandomState(0)
        scores = DCI.compute_dci(dataloader, representation_function, 1000, 1000, 100)
        self.assertTrue(0.9 <= scores["disentanglement"] <= 1.0)
        target = 1. - np.log(2) / np.log(10)
        self.assertBetween(target - .1 <= scores["completeness"]<= target + .1)


if __name__ == '__main__':
    unittest.main()
