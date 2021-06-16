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
"""Tests for irs.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import unittest
from . import utils, IRS
import numpy as np
import torch
from torch.utils.data import DataLoader


def _identity_discretizer(target, num_bins):
    del num_bins
    return target


class IrsTest(unittest.TestCase):

    def test_metric(self):

        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: x
        scores = IRS.compute_irs(dataloader,
                                 representation_function,
                                 0.99, 3000, 100, 20)
        self.assertTrue(0.9 <= scores["IRS"] <= 1.0)

    def test_bad_metric(self):

        dataset = utils.IdentityObservationsData()
        dataloader = DataLoader(dataset, batch_size=100, shuffle=False)
        representation_function = lambda x: torch.zeros_like(x, dtype=torch.float64)
        scores = IRS.compute_irs(dataloader,
                                 representation_function,
                                 0.99, 3000, 100, 20)
        self.assertTrue(0.0 <= scores["IRS"] <= 0.1)

    def test_drop_constant_dims(self):
        ys = np.random.normal(0.0, 1.0, (100, 100))
        ys[0, :] = 1.
        ys[-1, :] = 0.
        active_ys = IRS._drop_constant_dims(ys)
        np.testing.assert_array_equal(active_ys, ys[1:-1])


if __name__ == "__main__":
    unittest.main()
