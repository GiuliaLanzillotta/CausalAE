# SAE experiment manager
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import ESAE, models_switch
from experiments.data import DatasetLoader
from visualisations import ModelVisualiser, SynthVecDataVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer, ModelDisentanglementEvaluator, LatentOrthogonalityEvaluator
from torchsummary import summary

class BaseVisualExperiment(pl.LightningModule):

    def __init__(self, params: dict, model, loader:DatasetLoader, verbose=True) -> None:
        super(BaseVisualExperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = loader
        self.model = model
        if verbose: self.print_model()
        self.visualiser = ModelVisualiser(self.model,
                                          self.loader.test,
                                          **params["vis_params"])
        self.val_every = self.params["trainer_params"]["val_check_interval"]
        self.plot_every = self.params['vis_params']['plot_every']
        self.log_every = self.params['logging_params']['log_every']
        self.num_FID_steps = len(self.val_dataloader())//20 # basically take 5% of the batches available
        self.num_val_steps = 0 #counts number of validation steps done

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def print_model(self):
        print("MODEL SUMMARY")
        summary(self.model.cuda(), (self.loader.data_shape))

    def make_plots(self, hybrids=True, originals=False, distortion=True):
        """originals: bool = Whether to plot the originals samples from the test set"""
        logger = self.logger.experiment
        figure = self.visualiser.plot_reconstructions(device=self.device)
        logger.add_figure("reconstructions", figure, global_step=self.global_step)

        try:
            figure = self.visualiser.plot_samples_from_prior(device=self.device)
            logger.add_figure("prior_samples", figure, global_step=self.global_step)
        except ValueError:pass #no prior samples stored yet

        figure = self.visualiser.plot_latent_traversals(device=self.device, tailored=True)
        logger.add_figure("traversals", figure, global_step=self.global_step)

        if hybrids: #plot the result of hybridisation in the latent space
            N=2
            figure = self.visualiser.plot_hybridisation(device=self.device, first=True)
            logger.add_figure(f"Hybridisation of {N+1} inputs on first dimensions.", figure, global_step=self.global_step)
            figure = self.visualiser.plot_hybridisation(device=self.device, first=False)
            logger.add_figure(f"Hybridisation of {N+1} inputs on last dimensions.", figure, global_step=self.global_step)

        if originals: # print the originals
            figure = self.visualiser.plot_originals()
            logger.add_figure("originals", figure)

        if distortion:
            figure = self.visualiser.plot_loss2distortion(device=self.device)
            logger.add_figure("Output-Latent distortion plot", figure, global_step=self.global_step)

        if isinstance(self.model, ESAE):
            figures = self.visualiser.plot_samples_controlled_hybridisation(device=self.device)
            for l,figure in enumerate(figures):
                logger.add_figure(f"prior_samples_hybridisation_level_{l}", figure, global_step=self.global_step)

    def validation_epoch_end(self, outputs):
        # Logging hyperparameters
        # Visualisation
        if self.num_val_steps%self.plot_every==0 or \
                self.global_step==self.params["trainer_params"]["max_steps"]:
            self.make_plots(originals=self.global_step==0)
        self.num_val_steps+=1

    def test_epoch_end(self, outputs):
        self.make_plots(originals=self.global_step==0)


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


class BaseVecExperiment(pl.LightningModule):

    def __init__(self, params: dict, verbose=True) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        super(BaseVecExperiment, self).__init__()
        self.params = params
        self.loader = DatasetLoader(params["data_params"])
        self.dim_in = self.loader.data_shape # C, H, W
        self.model = models_switch[params["model_params"]["name"]](params["model_params"], self.dim_in, **params["model_params"])
        if verbose: self.print_model()
        self.model_visualiser = ModelVisualiser(self.model,
                                                self.loader.test,
                                                **params["vis_params"])
        self.data_visualiser = SynthVecDataVisualiser(self.loader.test)
        self.val_every = self.params["trainer_params"]["val_check_interval"]
        self.plot_every = self.params['vis_params']['plot_every']
        self.num_val_steps = 0 #counts number of validation steps done
        self.log_every = self.params['logging_params']['log_every']


    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def print_model(self):
        summary(self.model.cuda(), (self.loader.data_shape))

    def make_plots(self, dataset=False, distortion=True):
        """originals: bool = Whether to plot the originals samples from the test set"""
        logger = self.logger.experiment

        if dataset: # print the originals
            graph =  self.data_visualiser.plot_graph()
            logger.add_figure("graph", graph)
            noises =  self.data_visualiser.plot_noises_distributions()
            logger.add_figure("noises' distribution", noises)
            causes2noises =  self.data_visualiser.plot_causes2noises()
            logger.add_figure("cause to noise plot", causes2noises)

        if distortion:
            figure = self.model_visualiser.plot_loss2distortion(device=self.device)
            logger.add_figure("Output-Latent distortion plot", figure)

    def configure_optimizers(self):
        opt_params = self.params["opt_params"]
        optimizer = optim.Adam(self.model.parameters(),
                               lr=opt_params['LR'],
                               weight_decay=opt_params['weight_decay'])
        return optimizer

    def train_dataloader(self):
        return self.loader.train

    def val_dataloader(self):
        return self.loader.val

    def test_dataloader(self):
        return self.loader.test

