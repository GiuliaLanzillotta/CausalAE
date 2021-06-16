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
"""Tests for sap_score.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
from . import utils, SAP
import numpy as np
import torch
from torch.utils.data import DataLoader

class SapScoreTest(unittest.TestCase):

    def test_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: x
        scores = SAP.compute_sap(dataloader, representation_function,
                                 3000, 3000, 100, continuous_factors=True)
        self.assertTrue(0.9 <= scores["SAP_score"] <= 1.0)

    def test_bad_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: torch.zeros_like(x, dtype=torch.float64)
        scores = SAP.compute_sap(dataloader, representation_function,
                                 3000, 3000, 100, continuous_factors=True)
        self.assertTrue(0.0 <= scores["SAP_score"] <= 0.2)

    def test_duplicated_latent_space(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)

        def representation_function(x):
            return torch.hstack([x, x])

        scores = SAP.compute_sap(dataloader, representation_function,
                                 3000, 3000, 100, continuous_factors=True)
        self.assertTrue(0.0 <= scores["SAP_score"] <= 0.2)


if __name__ == "__main__":
    unittest.main()
