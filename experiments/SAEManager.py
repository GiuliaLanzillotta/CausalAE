# SAE experiment manager 
import numpy as np
import torch
from matplotlib.lines import Line2D
from torch import Tensor
from torch import optim
from models import SAE
from experiments.data import DatasetLoader
from torchvision import utils as tvu
import pytorch_lightning as pl
import matplotlib.pyplot as plt

class SAEXperiment(pl.LightningModule):

    def __init__(self, params: dict) -> None:
        super(SAEXperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = DatasetLoader(params["data_params"])
        dim_in =  self.loader.data_shape # C, H, W
        self.model = SAE(params["model_params"], dim_in)
        print(self.model)
        self.burn_in = params["opt_params"]["auto_epochs"]
        # For tensorboard logging (saving the graph)
        self.example_input_array = torch.rand((1,) + self.loader.data_shape, requires_grad=False)

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        mode="auto" if self.current_epoch<=self.burn_in else "hybrid"
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs, mode=mode)
        BCE, FID, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE', MSE, prog_bar=True, on_epoch=True, on_step=True)
        # TODO: include FID
        # self.log('FID', FID, prog_bar=True, on_epoch=True, on_step=True)
        return BCE

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch%self.params['vis_params']['plot_every']==0:
            self.plot_grad_flow(self.model.named_parameters())

    def validation_step(self, batch, batch_idx):
        mode="auto" if self.current_epoch<=self.burn_in else "hybrid"
        input_imgs, labels = batch
        X_hat = self.forward(input_imgs, mode=mode)
        BCE, FID, MSE = self.model.loss_function(X_hat, input_imgs)# Logging
        self.log('BCE_valid', BCE, prog_bar=True, on_epoch=True, on_step=True)
        self.log('MSE_valid', MSE, prog_bar=True, on_epoch=True, on_step=True)
        # TODO: include FID
        # self.log('FID', FID, prog_bar=True, on_epoch=True, on_step=True)
        return BCE

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor(outputs).mean()
        if self.current_epoch%self.params['vis_params']['plot_every']==0:
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


    def plot_grad_flow(self, named_parameters):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads= []
        layers = []
        for n, p in named_parameters:
            if(p.requires_grad) and ("bias" not in n):
                try:
                    layers.append(n)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                except AttributeError:
                    print(n+" has no gradient. Skipping")
                    continue
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
        plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
        plt.plot(ave_grads, alpha=0.3, color="b")
        plt.plot(max_grads, alpha=0.3, color="c")
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
        plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        plt.tick_params(axis='both', labelsize=4)
        plt.xlim(left=0, right=len(ave_grads))
        plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
        plt.xlabel("Layers")
        plt.ylabel("average gradient")
        plt.title("Gradient flow")
        plt.grid(True)
        plt.tight_layout()
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        folder=f"./{self.logger.save_dir}{self.logger.name}/{self.logger.version}/"
        plt.savefig(f"{folder}gradient_{self.logger.name}_{self.current_epoch}.png", dpi=200)

    def sample_images(self):
        """ Take a batch of images from the validation set and plot their reconstruction.
        Note: this function is called for each epoch"""
        # Get sample reconstruction image
        device = self.device
        test_input, test_label = next(iter(self.test_dataloader()))
        mode="auto" if self.current_epoch<=self.burn_in else "hybrid"
        recons = self.model.forward(test_input.to(device),mode=mode)
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
