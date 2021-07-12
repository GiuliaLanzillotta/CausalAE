"""Experient manager for the E(quivariant)-SAE model"""
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import ESAE
from experiments.data import DatasetLoader
from experiments.BaseManager import BaseVisualExperiment, BaseVecExperiment
from visualisations import ModelVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer

# Warmup schemes that should mimic the beta ones for the partial sampling



class ESAEXperiment(BaseVisualExperiment):

    def __init__(self, params: dict, verbose=True) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = ESAE(params["model_params"], dim_in)
        super(ESAEXperiment, self).__init__(params, model, loader, verbose=verbose)
        self.use_MSE = params["model_params"]["loss_type"] == "MSE"

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=False)
        losses = self.model.loss_function(*results, X = input_imgs,
                                          lamda = self.model.params["lamda"],
                                          device=self.device,
                                          use_MSE=self.use_MSE)
        # Logging
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log('REC_loss', losses["Reconstruction_loss"], on_epoch=True)
        self.log('REG_loss', losses["Regularization_loss"], on_epoch=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        val_losses = self.model.loss_function(*results, X = input_imgs,
                                              lamda = self.model.params["lamda"],
                                              device=self.device,
                                              use_MSE=self.use_MSE)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return val_losses

    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        test_losses = self.model.loss_function(*results, X = input_imgs,
                                               lamda = self.model.params["lamda"],
                                               device=self.device,
                                               use_MSE=self.use_MSE)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', test_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)


class ESAEVecExperiment(BaseVecExperiment):

    def __init__(self, params: dict, verbose=True) -> None:
        super(ESAEVecExperiment, self).__init__(params, verbose=verbose)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=False)
        losses = self.model.loss_function(*results, X = inputs,
                                          lamda = self.model.params["lamda"],
                                          device=self.device)
        # Logging
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log('REC_loss', losses["Reconstruction_loss"], on_epoch=True)
        self.log('REG_loss', losses["Regularization_loss"], on_epoch=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.model_visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=True)
        val_losses = self.model.loss_function(*results, X = inputs,
                                              lamda = self.model.params["lamda"],
                                              device=self.device)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return val_losses

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=True)
        test_losses = self.model.loss_function(*results, X = inputs,
                                               lamda = self.model.params["lamda"],
                                               device=self.device)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', test_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)

