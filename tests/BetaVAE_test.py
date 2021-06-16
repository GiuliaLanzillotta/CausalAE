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
"""Tests for beta_vae.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from . import utils
from . import BetaVAE


class BetaVaeTest(unittest.TestCase):

    def test_metric(self):
        dataset = utils.IdentityObservationsData()
        representation_function = lambda x: x
        scores = BetaVAE.compute_beta_vae_sklearn(dataset,
                                                  representation_function,
                                                  16, 2000, 2000)
        self.assertTrue(0.9 <= scores["train_accuracy"] <= 1.0)
        self.assertTrue(0.9 <= scores["eval_accuracy"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
