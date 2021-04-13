# SAE experiment manager 
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import SAE
from experiments.data import DatasetLoader
from torchvision import utils as tvu
import pytorch_lightning as pl

class SAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(SAEXperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = DatasetLoader(params["data_params"])
        dim_in =  self.loader.data_shape # C, H, W
        self.model = SAE(params["model_params"], dim_in, self.device)
        # For tensorboard logging (saving the graph)
        self.example_input_array = torch.rand((1,) + self.loader.data_shape, requires_grad=False)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, FID = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE', BCE, prog_bar=True, on_epoch=True, on_step=True)
        # TODO: include FID
        # self.log('FID', FID, prog_bar=True, on_epoch=True, on_step=True)
        return BCE

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs)
        BCE, FID = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        # TODO: include FID
        # self.log('FID', FID, prog_bar=True, on_epoch=True, on_step=True)
        return BCE

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor(outputs).mean()
        if self.current_epoch%self.params['logging_params']['plot_every']==0:
            self.sample_images() # save images every plot_every_epochs epochs
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

    def sample_images(self):
        """ Take a batch of images from the validation set and plot their reconstruction.
        Note: this function is called for each epoch"""
        # Get sample reconstruction image
        device = self.device
        test_input, test_label = next(iter(self.test_dataloader()))
        recons = self.model.generate(test_input.to(device))
        folder=f"./{self.logger.save_dir}{self.logger.name}/{self.logger.version}/"
        # save originals at the beginning
        if self.current_epoch==0:
            tvu.save_image(test_input,
                           fp= f"{folder}original_{self.logger.name}.png",
                           normalize=True,
                           nrow=int(np.sqrt(self.params["data_params"]["batch_size"]))) # plot a square grid
        tvu.save_image(recons.data,
                       fp= f"{folder}recons_{self.logger.name}_{self.current_epoch}.png",
                       normalize=True,
                       nrow=int(np.sqrt(self.params["data_params"]["batch_size"]))) # plot a square grid
        # clean
        del test_input, recons

    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test
