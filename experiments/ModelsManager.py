""" Manager of all models - takes care of trainin/validation/test"""

from experiments.BaseManager import BaseVisualExperiment
from experiments.data import DatasetLoader
from models import models_switch
from .utils import SchedulersManager
import torch

torch.autograd.set_detect_anomaly(True)

class GenerativeAEExperiment(BaseVisualExperiment):
    """ Manager of all autoencoder experiments
    The following conditions have to be verified:
    Basically all models implementing a GenerativeAE can be managed with this class
    """

    def __init__(self, params: dict, verbose=True) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        params['model_params']['dim_in'] = dim_in
        model = models_switch[params["model_params"]["name"]](params["model_params"])
        super(GenerativeAEExperiment, self).__init__(params, model, loader, verbose=verbose)
        loss_type = params["model_params"]["loss_type"]
        assert loss_type in ["MSE","BCE"], "Requested loss type not available"
        self.use_MSE = loss_type == "MSE"
        self.scheduler_manager = SchedulersManager(params["model_params"]["name"], params)
        self.params['model_params'].update(self.scheduler_manager.weights)
        self.scheduler_step = params['opt_params']['schedulers_step']

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        if self.global_step%self.scheduler_step == 0:
            self.scheduler_manager.update_weights(model=self.model, step_num=self.global_step)
            self.params['model_params'].update(self.scheduler_manager.weights)
        results = self.forward(input_imgs, update_prior=True, integrate=False)
        losses = self.model.loss_function(*results,
                                          X=input_imgs,
                                          device=self.device,
                                          use_MSE=self.use_MSE,
                                          **self.params['model_params'],
                                          **self.params['opt_params'])
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in losses.items()})
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        losses = self.model.loss_function(*results,
                                          X=input_imgs,
                                          device=self.device,
                                          use_MSE=self.use_MSE,
                                          **self.params['model_params'],
                                           **self.params['opt_params'])
        self.log('val_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in losses.items()})
        return losses

    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        losses = self.model.loss_function(*results,
                                          X=input_imgs,
                                          device=self.device,
                                          use_MSE=self.use_MSE,
                                          **self.params['model_params'],
                                          **self.params['opt_params'])
        self.log('test_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in losses.items()})
        return losses
