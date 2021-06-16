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
"""Tests for mig.py."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
from . import utils, MIG
import numpy as np
import torch
from torch.utils.data import DataLoader



def _identity_discretizer(target, num_bins):
    del num_bins
    return target

class MIGTest(unittest.TestCase):

    def test_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: x
        scores = MIG.compute_mig(dataloader,
                                 representation_function,
                                 3000, 100, 16)
        self.assertTrue(0.9 <= scores["discrete_mig"] <= 1.0)

    def test_bad_metric(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = torch.zeros_like
        scores = MIG.compute_mig(dataloader,
                                 representation_function,
                                 3000, 100, 16)
        self.assertTrue(0.0 <= scores["discrete_mig"] <= 0.2)

    def test_duplicated_latent_space(self):
        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        def representation_function(x):
            return torch.hstack([x, x])

        scores = MIG.compute_mig(dataloader,
                                 representation_function,
                                 3000, 100, 16)
        self.assertTrue(0.0 <= scores["discrete_mig"] <= 0.1)


if __name__ == "__main__":
    unittest.main()
