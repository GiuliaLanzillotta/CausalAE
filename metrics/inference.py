"""Evaluates given representation on inference task"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import time

import numpy as np
import torch
from absl import logging
from torch import optim
from torch.utils.data import DataLoader
from torchsummary import summary

from .DCI import compute_importance_gbt
from models import INET
from . import utils


def do_inference(loader_train:DataLoader,
                 loader_test:DataLoader,
                 representation_function,
                 device:str,
                 num_train=20000,
                 num_test=10000,
                 batch_size=16):
    """Trains, stores and evaluate an inference model

    Args:
      loader_train: data to be sampled from to build training dataset
      loader_test: data to be sampled from to build test dataset
      representation_function: Function that takes observations as input and
        outputs a dim_representation sized representation for each observation.
      num_train: Number of points used for training.
      num_test: Number of points used for testing.
      batch_size: Batch size for sampling.

    Returns:
        importance_matrix, train_acc, test_acc (all np.ndarrays)
    """
    # mus_train are of shape [num_codes, num_train], while ys_train are of shape
    # [num_factors, num_train].
    logging.info(f"Generating training set with {num_train} samples.")
    mus_train, ys_train = utils.generate_batch_factor_code(loader_train, representation_function, num_train, batch_size, device=device)
    assert mus_train.shape[1] == num_train; assert ys_train.shape[1] == num_train
    logging.info(f"Generating testing set with {num_test} samples.")
    mus_test, ys_test = utils.generate_batch_factor_code(loader_test, representation_function, num_test, batch_size, device=device) #TODO: maybe factorise this duplicated code
    assert mus_test.shape[1] == num_test; assert ys_test.shape[1] == num_test

    print("Starting training inference model...")
    start = time.time()
    # Compute inference scores on train and test
    importance_matrix, train_acc, test_acc = compute_importance_gbt(mus_train, ys_train, mus_test, ys_test)
    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Average test Acc: {:4f}'.format(np.mean(test_acc)))
    return importance_matrix, train_acc, test_acc
