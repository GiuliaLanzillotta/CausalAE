import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import VAE
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
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
        self.KL_weight = self.loader.num_samples/self.params['data_params']['batch_size']
        # For tensorboard logging (saving the graph)
        self.example_input_array = torch.rand((1,) + self.loader.data_shape)


    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        if self.current_epoch%10==0 and self.current_epoch!=0:
            self.KL_weight *= self.params['opt_params']["KL_decay"] # decaying the KL term
        train_loss = self.model.loss_function(*results,
                                              X = input_imgs,
                                              KL_weight =  self.KL_weight)
        # Logging
        self.log('train_loss', train_loss["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in train_loss.items()})

        return train_loss["loss"]

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch%self.params['logging_params']['plot_every']==0:
            self.plot_grad_flow(self.model.named_parameters())

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs)
        val_loss = self.model.loss_function(*results,
                                            X = input_imgs,
                                            KL_weight =  self.KL_weight)
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_loss["loss"], prog_bar=True, logger=True)
        return val_loss

    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in outputs]).mean()
        if self.current_epoch%self.params['logging_params']['plot_every']==0:
            self.sample_images() # save images every plot_every_epochs epochs
        return {"val_loss":avg_val_loss}

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

        # Get randomly sampled images
        samples = self.model.generate_standard(self.params["data_params"]["batch_size"], device=self.device)
        tvu.save_image(samples.cpu().data,
                       fp=f"{folder}samples_{self.logger.name}_{self.current_epoch}.png",
                       normalize=True,
                       nrow=int(np.sqrt(self.params["data_params"]["batch_size"])))
        # clean
        del test_input, recons, samples

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
                layers.append(n)
                try:
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


    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test
