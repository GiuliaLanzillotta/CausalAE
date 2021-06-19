"""Experient manager for the E(quivariant)-SAE model"""
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import SAE
from experiments.data import DatasetLoader
from experiments.BaseManager import BaseExperiment
from visualisations import ModelVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer

# Warmup schemes that should mimic the beta ones for the partial sampling



class ESAEXperiment(BaseExperiment):

    def __init__(self, params: dict) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = SAE(params["model_params"], dim_in)
        super(ESAEXperiment, self).__init__(params, model, loader)

    def training_step(self, batch, batch_idx):
        #TODO

    def score_FID(self, batch_idx, inputs, results):
        #TODO

    def validation_step(self, batch, batch_idx):
        #TODO

    def test_step(self, batch, batch_idx):
        #TODO
