"""Manager of models with explicit causal structure in the latent space (X-class)"""
from typing import List

from experiments.BaseManager import BaseVisualExperiment, BaseVecExperiment
from experiments.data import DatasetLoader
from models import models_switch


class XExperiment(BaseVisualExperiment):
    """
        Class of experiments for all the models that satisfy these characteristics:
        - constructor that accepts (params, dim_in)
        - loss that accepts (*results, X=, lamda=, device=, use_MSE=)

    Basically all the models implementing X- classes
    """

    def __init__(self, params: dict, verbose=True) -> None:
        # When initialised the dataset loader will download or load the data from the folder
        # split in train/test, apply transformations, divide in batches, extract data dimension
        loader = DatasetLoader(params["data_params"])
        dim_in =  loader.data_shape # C, H, W
        # only models that have
        model = models_switch[params["model_params"]["name"]](params["model_params"], dim_in)
        super(XExperiment, self).__init__(params, model, loader, verbose=verbose)
        self.use_MSE = params["model_params"]["loss_type"] == "MSE"

    def training_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=False)
        if not isinstance(results, List): results = [results]
        losses = self.model.loss_function(*results,
                                          X=input_imgs,
                                          device=self.device,
                                          use_MSE=self.use_MSE,
                                          **self.params['model_params'],
                                          **self.params['opt_params'])
        if self.model.sparsity_on:
            sparsity_penalty = self.model.caual_block.masks_sparsity_penalty()
            losses['sparsity_penalty'] = sparsity_penalty
            losses['loss'] += sparsity_penalty
        # Logging
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log_dict({key: val.item() for key, val in losses.items()})
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def validation_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        val_losses = self.model.loss_function(*results, X = input_imgs,
                                              device=self.device,
                                              use_MSE=self.use_MSE,
                                              **self.params['model_params'],
                                              **self.params['opt_params'])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict({key: val.item() for key, val in val_losses.items()})
        return val_losses

    def test_step(self, batch, batch_idx):
        input_imgs, labels = batch
        results = self.forward(input_imgs, update_prior=True, integrate=True)
        test_losses = self.model.loss_function(*results, X = input_imgs,
                                               device=self.device,
                                               use_MSE=self.use_MSE,
                                               **self.params['model_params'],
                                               **self.params['opt_params'])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', test_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict({key: val.item() for key, val in test_losses.items()})


class RegVecExperiment(BaseVecExperiment):

    def __init__(self, params: dict, verbose=True) -> None:
        super(RegVecExperiment, self).__init__(params, verbose=verbose)

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=False)
        losses = self.model.loss_function(*results, X = inputs,
                                          device=self.device,
                                          **self.params['model_params'],
                                          **self.params['opt_params'])
        # Logging
        self.log('train_loss', losses["loss"], prog_bar=True, on_epoch=True, on_step=True)
        self.log('REC_loss', losses["Reconstruction_loss"], on_epoch=True)
        self.log('REG_loss', losses["Regularization_loss"], on_epoch=True)
        if self.global_step%(self.plot_every*self.val_every)==0 and self.global_step>0:
            figure = self.model_visualiser.plot_training_gradients()
            self.logger.experiment.add_figure("gradient", figure, global_step=self.global_step)
        return losses

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=True)
        val_losses = self.model.loss_function(*results, X = inputs,
                                              device=self.device,
                                              **self.params['model_params'],
                                              **self.params['opt_params'])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', val_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return val_losses

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        results = self.forward(inputs, update_prior=True, integrate=True)
        test_losses = self.model.loss_function(*results, X = inputs,
                                               device=self.device,
                                               **self.params['model_params'],
                                               **self.params['opt_params'])
        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', test_losses["loss"], prog_bar=True, logger=True, on_step=True, on_epoch=True)

