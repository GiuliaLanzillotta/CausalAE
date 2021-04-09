import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import VAE
from data import DatasetLoader
import pytorch_lightning as pl


class VAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(VAEXperiment, self).__init__()
        self.curr_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        # logging
        sampleImg=torch.rand((1,) + self.loader.data_shape)
        self.logger.log_graph(self.model, sampleImg)
        #TODO: merge all hyperparameters
        #TODO: add metrics
        self.logger.log_hyperparams(params["opt_params"])


    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              X = input_imgs,
                                              KL_weight =  self.KL_weight)
        # Logging
        tensorboard_logs = {"train_loss":train_loss["loss"]}
        self.log('train_loss', train_loss["loss"], prog_bar=True)
        #TODO: check what the logger does
        self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})

        return {"loss":train_loss["loss"], "log":tensorboard_logs}

    def training_step_end(self, *args, **kwargs):
        #TODO
        pass

    def training_epoch_end(self, outputs):

        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        for name,params in self.model.named_parameters():
            self.logger.experiment.add_histogram(name,params,self.current_epoch)
        self.logger.experiment.add_scalar('train_epoch_loss', avg_loss, self.current_epoch)
        return {'loss': avg_loss}

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        input_imgs, labels = batch
        results = self.forward(input_imgs, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            X = input_imgs,
                                            KL_weight =  self.KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True)
        return {"val_loss":val_loss["loss"]}

    def validation_step_end(self, *args, **kwargs):
        #TODO
        pass

    def validation_epoch_end(self, outputs):
        """ Outputs is of the form [loss_batch1, loss_batch2, ...]
        Each loss element is a dict with keys "loss", "Reconstruction_loss" and "KL"
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.sample_images()
        return {'val_loss': avg_loss}

    def test_step(self, *args, **kwargs):
        #TODO: include test step
        pass

    def configure_optimizers(self):
        opt_params = self.params["opt_params"]
        optimizer = optim.Adam(self.model.parameters(),
                               lr=opt_params['LR'],
                               weight_decay=opt_params['weight_decay'])

        if opt_params['scheduler_gamma'] is not None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma = self.params['scheduler_gamma'])
            return optimizer, scheduler

        return optimizer

    def sample_images(self):
        #TODO: fix sample_images

        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        recons = self.model.generate(test_input, labels = test_label)
        vutils.save_image(recons.data,
                          f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                          f"recons_{self.logger.name}_{self.current_epoch}.png",
                          normalize=True,
                          nrow=12)

        # vutils.save_image(test_input.data,
        #                   f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
        #                   f"real_img_{self.logger.name}_{self.current_epoch}.png",
        #                   normalize=True,
        #                   nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              f"{self.logger.save_dir}{self.logger.name}/version_{self.logger.version}/"
                              f"{self.logger.name}_{self.current_epoch}.png",
                              normalize=True,
                              nrow=12)
        except:
            pass


        del test_input, recons #, samples

    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test
