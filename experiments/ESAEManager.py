"""Experient manager for the E(quivariant)-SAE model"""
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import ESAE
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
        model = ESAE(params["model_params"], dim_in)
        super(ESAEXperiment, self).__init__(params, model, loader)

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        losses = self.model.loss_function(*results, X = input_imgs, lamda = self.params["lamda"])
        # Logging
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log('REC_loss', losses["Reconstruction_loss"], on_epoch=True)
        self.log('REG_loss', losses["Regularization_loss"], on_epoch=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients(self.global_step)
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def score_FID(self, batch_idx, inputs, results):
        if batch_idx==0:
            self._fidscorer.start_new_scoring(self.params['data_params']['batch_size']*self.num_FID_steps,device=self.device)
        if  batch_idx<=self.num_FID_steps:#only one every 50 batches is included to avoid memory issues
            try: self._fidscorer.get_activations(inputs, self.model.act(results[0])) #store activations for current batch
            except Exception as e:
                print(e)
                print("Reached the end of FID scorer buffer")

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        val_losses = self.model.loss_function(*results, X = input_imgs, lamda = self.params["lamda"])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if (self.num_val_steps)%(self.score_every)==0 and self.num_val_steps!=0:
            self.score_FID(batch_idx, input_imgs, results)
        return val_losses

    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        test_losses = self.model.loss_function(*results, X = input_imgs, lamda = self.params["lamda"])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', test_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.score_FID(batch_idx, input_imgs, results)

