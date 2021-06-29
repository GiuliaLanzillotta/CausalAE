# SAE experiment manager
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import BASE, ESAE
from experiments.data import DatasetLoader
from visualisations import ModelVisualiser
import pytorch_lightning as pl
from metrics import FIDScorer, ModelDisentanglementEvaluator


class BaseExperiment(pl.LightningModule):

    def __init__(self, params: dict, model:BASE, loader:DatasetLoader) -> None:
        super(BaseExperiment, self).__init__()
        self.params = params
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        self.loader = loader
        self.model = model
        self.visualiser = ModelVisualiser(self.model,
                                          self.loader.test,
                                          **params["vis_params"])
        self._fidscorer = FIDScorer()
        self._disentanglementScorer = ModelDisentanglementEvaluator(self.model, self.val_dataloader())
        self.val_every = self.params["trainer_params"]["val_check_interval"]
        self.plot_every = self.params['vis_params']['plot_every']
        self.score_every = self.params['logging_params']['score_every']
        self.log_every = self.params['logging_params']['log_every']
        self.FID_scoring = self.params['data_params']['FID_scoring']
        self.num_FID_steps = len(self.val_dataloader())//20 # basically take 5% of the batches available
        self.num_val_steps = 0 #counts number of validation steps done

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def make_plots(self, originals=False):
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

        if originals: # print the originals
            figure = self.visualiser.plot_originals()
            logger.add_figure("originals", figure)

        if isinstance(self.model, ESAE):
            figures = self.visualiser.plot_samples_controlled_hybridisation(device=self.device)
            for l,figure in enumerate(figures):
                logger.add_figure(f"prior_samples_hybridisation_level_{l}", figure, global_step=self.global_step)


    def validation_epoch_end(self, outputs):
        # Logging hyperparameters
        # Visualisation
        #TODO: insert disentanglement scoring once in a while
        #TODO: check plotting here
        if self.num_val_steps%self.plot_every==0 or \
                self.global_step==self.params["trainer_params"]["max_steps"]:
            self.make_plots(originals=self.global_step==0)
        # Scoring val performance
        if self.num_val_steps%self.score_every==0 and self.num_val_steps!=0:
            # compute and store the fid scoring
            disentanglement_scores, complete_scores = self._disentanglementScorer.score_model(device=self.device)
            for k,v in disentanglement_scores.items():
                self.log(k, v, prog_bar=False)
            if self.FID_scoring:
                fid_score = self._fidscorer.calculate_fid()
                self.log("FID", fid_score, prog_bar=True)

        self.num_val_steps+=1

    def test_epoch_end(self, outputs):
        disentanglement_scores, complete_scores = self._disentanglementScorer.score_model(betaVAE=False,device=self.device)
        for k,v in disentanglement_scores.items():
            self.log(k, v, prog_bar=False)
        _scores = disentanglement_scores
        if self.FID_scoring:
            fid_score = self._fidscorer.calculate_fid()
            self.log("FID_test", fid_score, prog_bar=False)
            _scores['FID'] = fid_score

        self.make_plots(originals=self.global_step==0)

        return _scores


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
