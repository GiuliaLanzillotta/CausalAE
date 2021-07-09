# SAE experiment manager 
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import SAE
from experiments.data import DatasetLoader
from experiments.BaseManager import BaseVisualExperiment, BaseVecExperiment
from visualisations import ModelVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer

# Warmup schemes that should mimic the beta ones for the partial sampling



class SAEXperiment(BaseVisualExperiment):

    def __init__(self, params: dict, verbose=True) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = SAE(params["model_params"], dim_in)
        super(SAEXperiment, self).__init__(params, model, loader, verbose=verbose)
        self.loss_type = params["model_params"]["loss_type"]
        assert self.loss_type in ["MSE","BCE"], "Requested loss type not available"

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE', MSE, prog_bar=True, on_epoch=True, on_step=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        if self.loss_type=="MSE":return MSE
        return BCE

    def score_FID(self, batch_idx, inputs, results):
        if batch_idx==0:
            self._fidscorer.start_new_scoring(self.params['data_params']['batch_size']*self.num_FID_steps,device=self.device)
        if  batch_idx<=self.num_FID_steps:#only one every 50 batches is included to avoid memory issues
            try: self._fidscorer.get_activations(inputs, self.model.act(results)) #store activations for current batch
            except Exception as e:
                print(e)
                print("Reached the end of FID scorer buffer")

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE_valid', MSE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_loss', BCE, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if self.num_val_steps%self.score_every==0 and self.num_val_steps!=0 and self.FID_scoring:
            self.score_FID(batch_idx, input_imgs, X_hat)
        if self.loss_type=="MSE":return MSE
        return BCE

    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_test', BCE, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('MSE_test', MSE, prog_bar=True,logger=True, on_step=True, on_epoch=True)
        if self.FID_scoring: self.score_FID(batch_idx, input_imgs, X_hat)
        if self.loss_type=="MSE":return MSE
        return BCE



class SAEVecExperiment(BaseVecExperiment):

    def __init__(self, params: dict, verbose=True) -> None:
        super(SAEVecExperiment, self).__init__(params, verbose=verbose)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        X_hat = self.forward(inputs)
        BCE, MSE = self.model.loss_function(X_hat, inputs)# Logging
        self.log('MSE', MSE, prog_bar=True, on_epoch=True, on_step=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.model_visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return MSE

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        X_hat = self.forward(inputs)
        BCE, MSE = self.model.loss_function(X_hat, inputs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE_valid', MSE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_loss', BCE, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return MSE

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        X_hat = self.forward(inputs)
        BCE, MSE = self.model.loss_function(X_hat, inputs)# Logging
        self.log('BCE_test', BCE, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('MSE_test', MSE, prog_bar=True,logger=True, on_step=True, on_epoch=True)
        return MSE