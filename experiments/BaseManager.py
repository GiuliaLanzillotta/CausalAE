# SAE experiment manager
import numpy as np
import torch
from torch import Tensor
from torch import optim
from models import BASE
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
        self.val_every = self.params["trainer_params"]["val_check_interval"]
        self.plot_every = self.params['vis_params']['plot_every']
        self.score_every = self.params['logging_params']['score_every']
        self.log_every = self.params['logging_params']['log_every']
        self.num_FID_steps = len(self.val_dataloader())//20 # basically take 5% of the batches available
        self.num_val_steps = 0 #counts number of validation steps done

    def forward(self, inputs: Tensor, **kwargs) -> Tensor:
        return self.model(inputs, **kwargs)

    def validation_epoch_end(self, outputs):
        # Logging hyperparameters
        # Visualisation
        if self.num_val_steps%self.plot_every==0 or \
                self.global_step==self.params["trainer_params"]["max_steps"]:
            self.visualiser.plot_reconstructions(self.logger.experiment, self.global_step, device=self.device)
            try: self.visualiser.plot_samples_from_prior(self.logger.experiment, self.global_step, device=self.device)
            except ValueError:pass #no prior samples stored yet
            self.visualiser.plot_latent_traversals(self.logger.experiment, self.global_step,
                                                   device=self.device, tailored=True)
        # Scoring val performance
        if self.num_val_steps%self.score_every==0 and self.num_val_steps!=0:
            # compute and store the fid scoring
            fid_score = self._fidscorer.calculate_fid()
            self.log("FID", fid_score, prog_bar=True)

        self.num_val_steps+=1

    def test_step_end(self, outputs):
        _disentanglementScorer = ModelDisentanglementEvaluator(self.model, self.val_dataloader())
        disentanglement_scores, complete_scores = _disentanglementScorer.score_model()
        for k,v in disentanglement_scores.items():
            self.log(k, v, prog_bar=False)
        self.visualiser.plot_reconstructions(self.logger.experiment, device=self.device)
        try: self.visualiser.plot_samples_from_prior(self.logger.experiment, device=self.device)
        except ValueError:pass #no prior sampls stored yet
        self.visualiser.plot_latent_traversals(self.logger.experiment, device=self.device)
        fid_score = self._fidscorer.calculate_fid()
        self.log("FID_test", fid_score, prog_bar=False)
        _scores = disentanglement_scores
        _scores['FID'] = fid_score
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
