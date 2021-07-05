"""Script managing evaluation and exploration of models"""
import torch
from experiments import experiments_switch
from experiments.data import DatasetLoader
from models import VAE, SAE, models_switch
from models.BASE import GenerativeAE
from pathlib import Path
from configs import get_config
import os
import glob
import json
from visualisations import ModelVisualiser
from metrics import FIDScorer, ModelDisentanglementEvaluator

class ModelHandler(object):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, model_name:str, model_version:str, data:str, data_version="", **kwargs):
        self.config = get_config(tuning=False, model_name=model_name, data=data,
                                 version=model_version, data_version=data_version)
        self.experiment = experiments_switch[model_name](self.config)
        self.model = self.experiment.model
        assert issubclass(type(self.model), GenerativeAE), "Selected model is not an instance of GenerativeAE. " \
                                                           "Can only score disentanglement against Generative AE networks."
        self.model.eval()
        self.dataloader = self.experiment.loader
        print(model_name+ " model loaded.")
        self.visualiser = None
        self.fidscorer = None
        self.num_FID_steps = kwargs.get("num_FID_steps",10)
        self.device = kwargs.get("device","cpu")

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
        except ValueError:
            print(f"No checkpoint available at {actual_checkpoint_path}. Cannot load trained weights.")

    def score_model(self, FID=False, disentanglement=False, save_scores=False):
        """Scores the model on the test set in loss and other terms selected"""
        scores = {}
        loader = self.dataloader.test
        if disentanglement:
            _disentanglementScorer = ModelDisentanglementEvaluator(self.model, loader)
            disentanglement_scores, complete_scores = _disentanglementScorer.score_model()
            scores.update(disentanglement_scores)
            scores["extra_disentanglement"] = complete_scores
        if FID:
            if self.fidscorer is None:
                self.fidscorer = FIDScorer()
                for idx in range(self.num_FID_steps):
                    if idx==0:
                        self.fidscorer.start_new_scoring(self.config['data_params']['batch_size']*self.num_FID_steps,device=self.device)
                    inputs, labels = next(iter(loader))
                    reconstructions = self.model.generate(inputs, activate=True)
                    try: self.fidscorer.get_activations(inputs, reconstructions) #store activations for current batch
                    except Exception: print("Reached the end of FID scorer buffer")
                FID_score = self.fidscorer.calculate_fid()
                scores["FID"]=FID_score
        if save_scores:
            base_path = Path(self.config['logging_params']['save_dir']) / \
                        self.config['logging_params']['name'] / \
                        self.config['logging_params']['version'] / "scoring.json"
            with open(base_path, 'w') as o:
                json.dump(scores, o)

        return scores

class VisualModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, model_name: str, model_version: str, data: str, **kwargs):
        super().__init__(model_name, model_version, data, **kwargs)

    def plot_model(self, do_originals=False, do_reconstructions=False,
                   do_random_samples=False, do_traversals=False):

        plots = {}
        if self.visualiser is None:
            self.visualiser = ModelVisualiser(self.model, self.dataloader.test, **self.config["vis_params"])
        if do_reconstructions:
            plots["reconstructions"] = self.visualiser.plot_reconstructions(device=self.device)
        if do_random_samples:
            try: plots["random_samples"] = self.visualiser.plot_samples_from_prior(device=self.device)
            except ValueError:pass #no prior samples stored yet
        if do_traversals:
            plots["traversals"] = self.visualiser.plot_latent_traversals(device=self.device, tailored=True)
        if do_originals: # print the originals
            plots["originals"] = self.visualiser.plot_originals()
        return plots


class VectorModelHandler(ModelHandler):
    """Offers a series of tools to inspect a given model."""
    def __init__(self, model_name: str, model_version: str, data: str, data_version:str, **kwargs):
        super().__init__(model_name, model_version, data, data_version=data_version, **kwargs)

    def plot_model(self):
        #TODO
        pass


if __name__ == '__main__':
    handler = ModelHandler(model_name="BaseSAE", model_version="v16", data="RFDh5")
    handler.load_checkpoint()
    scores = handler.score_model(FID=False, disentanglement=True)