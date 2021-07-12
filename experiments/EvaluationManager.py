"""Script managing evaluation and exploration of models"""
import pickle

import torch
from experiments import experiments_switch
from models import VAEBase, SAE, models_switch
from models.BASE import GenerativeAE
from pathlib import Path
from configs import get_config
from experiments.data import DatasetLoader
import os
import glob
import json
import time
from visualisations import ModelVisualiser, SynthVecDataVisualiser
from metrics import FIDScorer, ModelDisentanglementEvaluator, LatentOrthogonalityEvaluator
import pytorch_lightning as pl

from torchsummary import summary

""" Evaluation toolbox for GenerativeAE models"""

class ModelHandler(object):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, experiment:pl.LightningModule, **kwargs):
        self.config = experiment.params
        self.experiment = experiment
        self.model = self.experiment.model
        self.dataloader = self.experiment.loader
        self.model.eval()
        model_name = self.config["model_params"]["name"]
        print(model_name+ " model hanlder loaded.")
        self.visualiser = None
        self.fidscorer = None
        self._orthogonalityScorer = None
        self._disentanglementScorer = None
        self.device = next(self.model.parameters()).device
        #self.send_model_to(self.device)

    @classmethod
    def from_experiment(cls, experiment:pl.LightningModule, **kwargs):
        return cls(experiment, **kwargs)

    @classmethod
    def from_config(cls, model_name:str, model_version:str, data:str, data_version="", **kwargs):
        config = get_config(tuning=False, model_name=model_name, data=data,
                                 version=model_version, data_version=data_version)
        experiment = experiments_switch[model_name](config, verbose=kwargs.get("verbose"))
        return cls(experiment, **kwargs)

    def score_disentanglement(self, **kwargs):
        """Scoring model on disentanglement metrics"""
        scores = {}
        full = kwargs.get("full",False) # full report
        if self._disentanglementScorer is None:
            self._disentanglementScorer = ModelDisentanglementEvaluator(self.model, self.dataloader.val)
        disentanglement_scores, complete_scores = self._disentanglementScorer.score_model(device = self.device)
        scores.update(disentanglement_scores)
        if full: scores["extra_disentanglement"] = complete_scores
        return scores

    def score_orthogonality(self, **kwargs):
        """Scoring the model against orthogonality measures """
        if self._orthogonalityScorer is None:
            try: unit_dim = self.model.unit_dim
            except AttributeError: unit_dim=1
            self._orthogonalityScorer = LatentOrthogonalityEvaluator(self.model, self.dataloader.val,
                                                                     self.model.latent_size, unit_dim)
        ortho_scores = self._orthogonalityScorer.score_latent(device=self.device,
                                                              strict=kwargs.get("strict",False),
                                                              hierarchy=kwargs.get("hierarchy",False))
        return ortho_scores

    def score_FID(self, **kwargs):
        """Scoring the model reconstraction quality with FID"""
        num_FID_steps = kwargs.get("num_FID_steps",10)

        if self.fidscorer is None:
            self.fidscorer = FIDScorer()

        for idx in range(num_FID_steps):
            if idx==0:
                self.fidscorer.start_new_scoring(self.config['data_params']['batch_size']*num_FID_steps,
                                                 device=self.device)
            inputs, labels = next(iter(self.dataloader.val))
            #TODO: check whether we want to evaluate prior samples here
            reconstructions = self.model.generate(inputs, activate=True)
            try: self.fidscorer.get_activations(inputs, reconstructions) #store activations for current batch
            except Exception: print("Reached the end of FID scorer buffer")

        FID_score = self.fidscorer.calculate_fid()

        return FID_score

    def send_model_to(self, device:str):
        if device=="cpu": self.model.cpu()
        else: self.model.cuda()

    def load_batch(self, train=True, valid=False, test=False):
        if train: loader = self.dataloader.train
        elif valid: loader = self.dataloader.val
        elif test: loader = self.dataloader.test
        else: raise NotImplementedError #TODO: include multiple sets from RFD
        new_batch = next(iter(loader))
        return new_batch

    def list_available_checkpoints(self):
        base_path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version']
        checkpoint_path =  base_path / "checkpoints/"
        checkpoints = glob.glob(str(checkpoint_path) + "/*ckpt")
        print("Available checkpoints at "+ str(checkpoint_path)+" :")
        print(checkpoints)
        return checkpoints

    def load_checkpoint(self, name=None):
        #TODO: together with checkpoint load some training status file, old results etc
        base_path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version']
        checkpoint_path =  base_path / "checkpoints/"
        hparams_path = str(base_path)+"/hparams.yaml"
        actual_checkpoint_path = ""

        try:
            if name is None:
                actual_checkpoint_path = max(glob.glob(str(checkpoint_path) + "/*ckpt"), key=os.path.getctime)
                print("Loading latest checkpoint at "+actual_checkpoint_path+" .")
            else:
                actual_checkpoint_path = str(checkpoint_path) +"/"+ name + ".ckpt"
                print("Loading selected checkpoint at "+ actual_checkpoint_path)

            self.experiment = self.experiment.load_from_checkpoint(actual_checkpoint_path,
                                                                   hparams_file=hparams_path,
                                                                   device=self.device,
                                                                   strict=False,
                                                                   params=self.config)
            self.model = self.experiment.model
            self.model.eval()
            self.send_model_to(self.device)
        except ValueError:
            print(f"No checkpoint available at "+str(checkpoint_path)+". Cannot load trained weights.")

    def score_model(self, FID=False, disentanglement=False, orthogonality=False,
                    save_scores=False, **kwargs):
        """Scores the model on the test set in loss and other terms selected"""
        start=time.time()
        scores = {}
        if orthogonality:
            ortho_scores = self.score_orthogonality(**kwargs)
            scores.update(ortho_scores)
        if disentanglement:
            disentanglement_scores = self.score_disentanglement(**kwargs)
            scores.update(disentanglement_scores)
        if FID:
            scores["FID"]= self.score_FID(**kwargs)
        end = time.time()
        print("Time elapsed for scoring {:.0f}".format(end-start))

        if save_scores:
            name = kwargs.get("name","scoring")
            path = Path(self.config['logging_params']['save_dir']) / \
                        self.config['logging_params']['name'] / \
                        self.config['logging_params']['version'] / (name+".pkl")
            with open(path, 'wb') as o:
                pickle.dump(scores, o)
        return scores

    def load_scores(self, **kwargs):
        """Return saved scores dictionary if any"""

        name = kwargs.get("name","scoring")
        path = Path(self.config['logging_params']['save_dir']) / \
                    self.config['logging_params']['name'] / \
                    self.config['logging_params']['version'] / (name+".pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)


        print("No scores file found at "+str(path))


class VisualModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def plot_model(self, do_originals=False, do_reconstructions=False,
                   do_random_samples=False, do_traversals=False, do_hybrisation=False,
                   do_loss2distortion=False, **kwargs):

        plots = {}
        if self.visualiser is None:
            self.visualiser = ModelVisualiser(self.model, self.dataloader.test, **self.config["vis_params"])
        if do_reconstructions:
            plots["reconstructions"] = self.visualiser.plot_reconstructions(device=self.device, **kwargs)
        if do_random_samples:
            try: plots["random_samples"] = self.visualiser.plot_samples_from_prior(device=self.device, **kwargs)
            except ValueError:pass #no prior samples stored yet
        if do_traversals:
            plots["traversals"] = self.visualiser.plot_latent_traversals(device=self.device, tailored=True, **kwargs)
        if do_originals: # print the originals
            plots["originals"] = self.visualiser.plot_originals()
        if do_hybrisation: # print the originals
            plots["hybrids"] = self.visualiser.plot_hybridisation(device=self.device, **kwargs)
        if do_loss2distortion:
            plots["distortion"] = self.visualiser.plot_loss2distortion(device=self.device, **kwargs)

        return plots


class VectorModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, experiment:pl.LightningModule, **kwargs):
        super().__init__(experiment, **kwargs)
        self.model_visualiser = None
        self.data_visualiser = None

    def switch_labels_to_noises(self):
        print("Loading noises as labels.")
        self.config["data_params"]["noise"] = True
        self.dataloader = DatasetLoader(self.config["data_params"])

    def plot_data(self):
        plots = {}
        if self.data_visualiser is None:
            self.data_visualiser = SynthVecDataVisualiser(self.dataloader.test)

        plots["graph"] = self.data_visualiser.plot_graph()
        plots["noises"] = self.data_visualiser.plot_noises_distributions()
        plots["causes2noises"] = self.data_visualiser.plot_causes2noises()

        return plots

    def plot_model(self, **kwargs):

        plots = {}
        if self.model_visualiser is None:
            self.model_visualiser = ModelVisualiser(self.model,
                                                    self.dataloader.test)

        figure = self.model_visualiser.plot_loss2distortion(device=self.device, **kwargs)
        plots["distortion"] = figure

        return plots



if __name__ == '__main__':
    handler = VectorModelHandler(model_name="VecESAE", model_version="standard", data="SynthVec", data_version="big", verbose=False)
    train = handler.dataloader.train.dataset.dataset
    print(train)

