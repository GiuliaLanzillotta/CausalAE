import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import VAE
from experiments.data import DatasetLoader
from torchvision import utils as tvu
import pytorch_lightning as pl


class VAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = DatasetLoader(params["data_params"])
        dim_in =  self.loader.data_shape # C, H, W
        self.model = VAE(params["model_params"], dim_in)
        # Additional initialisations (used in training and validation steps)
        N = int(np.product(self.loader.data_shape))
        M = self.model.latent_dim
        self.KL_weight = N/M


    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        train_loss = self.model.loss_function(*results,
                                              X = input_imgs,
                                              KL_weight =  self.KL_weight)
        # Logging
        tensorboard_logs = {"train_loss":train_loss["loss"]}
        self.log('train_loss', train_loss["loss"], prog_bar=True, on_epoch=True)
        self.log_dict({key: val.item() for key, val in train_loss.items()})

        return {"loss":train_loss["loss"], "log":tensorboard_logs}
    """
    def training_step_end(self, *args, **kwargs):
        #TODO
        pass

    def to_log_once(self):
        # logging
        sampleImg=torch.rand((1,) + self.loader.data_shape)
        self.logger.experiment.log_graph(self.model, sampleImg)
        #TODO: merge all hyperparameters
        #TODO: add metrics
        self.logger.experiment.log_hyperparams(self.params["opt_params"])

    def training_epoch_end(self, outputs):
        if self.current_epoch==1: self.to_log_once()
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        for name,params in self.model.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
        self.logger.experiment.add_scalar('train_epoch_loss', avg_loss, self.current_epoch)
        return {'loss': avg_loss}
    """

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        val_loss = self.model.loss_function(*results,
                                            X = input_imgs,
                                            KL_weight =  self.KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True, logger=True, on_epoch=True)

    """
    def validation_step_end(self, *args, **kwargs):
        #TODO
        pass

    def validation_epoch_end(self, outputs):
        #Outputs is of the form [loss_batch1, loss_batch2, ...]
        #Each loss element is a dict with keys "loss", "Reconstruction_loss" and "KL"
        
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.logger.experiment.add_scalar('val_epoch_loss', avg_loss, self.current_epoch)
        self.sample_images()
        return {'val_loss': avg_loss}
    """
    def test_step(self, *args, **kwargs):
        #TODO
        pass

    def configure_optimizers(self):
        opt_params = self.params["opt_params"]
        optimizer = optim.Adam(self.model.parameters(),
                               lr=opt_params['LR'],
                               weight_decay=opt_params['weight_decay'])

        #TODO: fix here
        """
        if opt_params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = opt_params['scheduler_gamma'])
            return optimizer, scheduler
        """
        return optimizer

    def sample_images(self):
        """ Take a batch of images from the validation set and plot their reconstruction.
        Note: this function is called for each epoch"""

        # Get sample reconstruction image
        test_input, test_label = next(iter(self.val_dataloader()))
        recons = self.model.generate(test_input)
        tvu.save_image(recons.data,
                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                       f"recons_{self.logger.name}_{self.current_epoch}.png",
                       normalize=True,
                       nrow=int(np.sqrt(self.params["data_params"]["batch_size"]))) # plot a square grid

        # Get randomly sampled images
        samples = self.model.sample_standard(self.params["data_params"]["batch_size"])
        tvu.save_image(samples.cpu().data,
                       f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                       f"{self.logger.name}_{self.current_epoch}.png",
                       normalize=True,
                       nrow=int(np.sqrt(self.params["data_params"]["batch_size"])))
        # clean
        del test_input, recons, samples

    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test
