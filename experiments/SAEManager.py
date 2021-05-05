# SAE experiment manager 
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


class SAEXperiment(BaseExperiment):

    def __init__(self, params: dict) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        model = SAE(params["model_params"], dim_in)
        super(SAEXperiment, self).__init__(params, model, loader)
        self.burn_in = params["opt_params"]["auto_steps"]
        if self.burn_in>=self.global_step:self.model.mode="hybrid"

    def training_step(self, batch, batch_idx):
        if self.global_step==self.burn_in: self.model.mode="hybrid"
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE', MSE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('step', self.global_step, prog_bar=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            self.visualiser.plot_training_gradients(self.global_step)
        return BCE

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE_valid', MSE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_loss', BCE, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if (self.num_val_steps)%(self.score_every)==0 and self.num_val_steps!=0:
            if batch_idx==0:
                self._fidscorer.start_new_scoring(self.params['data_params']['batch_size']*self.num_FID_steps,device=self.device)
            if  batch_idx<=self.num_FID_steps:#only one every 50 batches is included to avoid memory issues
                try: self._fidscorer.get_activations(input_imgs, self.model.act(X_hat)) #store activations for current batch
                except: print(self._fidscorer.start_idx)
        return BCE
