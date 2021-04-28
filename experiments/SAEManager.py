# SAE experiment manager 
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import SAE
from experiments.data import DatasetLoader
from visualisations import ModelVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer


class SAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(SAEXperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = DatasetLoader(params["data_params"])
        dim_in =  self.loader.data_shape # C, H, W
        self.model = SAE(params["model_params"], dim_in)
        self.visualiser = ModelVisualiser(self.model,
                                          params["logging_params"]["name"],
                                          params["logging_params"]["version"],
                                          self.loader.test,
                                          **params["vis_params"])
        self._fidscorer = FIDScorer()
        self.burn_in = params["opt_params"]["auto_epochs"]
        self.model.mode="auto" if self.burn_in>0 else "hybrid"
        # For tensorboard logging (saving the graph)
        self.example_input_array = torch.rand((1,) + self.loader.data_shape, requires_grad=False)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        if self.current_epoch==self.burn_in: self.model.mode="hybrid"
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, FID, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE', MSE, prog_bar=True, on_epoch=True, on_step=True)
        return BCE

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch%self.params['vis_params']['plot_every']==0:
            self.visualiser.plot_training_gradients(self.current_epoch)

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, FID, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE_valid', MSE, prog_bar=True, on_epoch=True, on_step=True)
        if self.current_epoch%self.params['logging_params']['score_every']==0 and self.current_epoch!=0:
            if batch_idx==0:# initialise the scoring for the current epoch
                self._fidscorer.start_new_scoring(
                    self.params['data_params']['batch_size']*len(self.val_dataloader())//20,
                    device=self.device)
            if batch_idx%20==0:#only one every 50 batches is included to avoid memory issues
                self._fidscorer.get_activations(input_imgs, self.model.act(X_hat)) #store activations for current batch
        return BCE

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor(outputs).mean()
        # Visualisation
        if self.current_epoch%self.params['vis_params']['plot_every']==0 or \
                self.current_epoch==self.params["trainer_params"]["max_epochs"]:
            self.visualiser.plot_reconstructions(self.current_epoch, device=self.device)
            try: self.visualiser.plot_samples_from_prior(self.current_epoch, device=self.device)
            except ValueError:pass #no prior samples stored yet
            self.visualiser.plot_latent_traversals(self.current_epoch, device=self.device)
        # Scoring val performance
        if self.current_epoch%self.params['logging_params']['score_every']==0 and self.current_epoch!=0:
            # compute and store the fid scoring
            fid_score = self._fidscorer.calculate_fid()
            self.log("FID", fid_score, prog_bar=True)
        self.log("val_loss",avg_val_loss, prog_bar=True)

    def test_step(self, *args, **kwargs):
        #TODO
        pass

    def configure_optimizers(self):
        opt_params = self.params["opt_params"]
        optimizer = optim.Adam(self.model.parameters(),
                               lr=opt_params['LR'],
                               weight_decay=opt_params['weight_decay'])
        """
        if opt_params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = opt_params['scheduler_gamma'])
            return optimizer
        """
        return optimizer


    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test
