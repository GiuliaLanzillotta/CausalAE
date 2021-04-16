""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import pytorch_lightning.utilities.seed
import yaml
import argparse
import numpy as np
import os
import glob
import torch
from experiments import experiments_switch
from configs import config_switch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune



def train_model(config:dict, tune:bool=False, num_epochs=10, num_gpus=0):
    """ Wrapper for the model training loop to be used by the hyper-parameter tuner"""
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'],
                                  version=config['logging_params']['version'],
                                  log_graph=False)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    pytorch_lightning.utilities.seed.seed_everything(config['logging_params']['manual_seed'])

    # callbacks
    callbacks = [ModelCheckpoint(monitor='val_loss', mode="min")]
    metrics = {"loss":"ptl/val_loss"}
    #after each validation epoch we report the above metric to Ray Tune
    if tune: callbacks.append(TuneReportCallback(metrics, on="validation_end"))

    # resuming from checkpoint
    checkpoint_path = os.path.join(config['logging_params']['save_dir'],
                                   config['logging_params']['name'],
                                   config['logging_params']['version'],
                                   "checkpoints/")
    latest_checkpoint = max(glob.glob(checkpoint_path + "\*ckpt"), key=os.path.getctime)
    runner = Trainer(min_epochs=1,
                     logger=tb_logger,
                     log_every_n_steps=50,
                     callbacks=callbacks,
                     progress_bar_refresh_rate=20,
                     checkpoint_callback=True,
                     resume_from_checkpoint=latest_checkpoint,
                     benchmark=False,
                     deterministic=True,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")

    experiment = experiments_switch[config['model_params']['name']](config)
    runner.fit(experiment)
    #todo save best model checkpoint path and metrics value

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAE experiments runner')
    parser.add_argument('--tune',  '-t',
                        dest="tune",
                        metavar='TUNE',
                        type=bool,
                        help =  'whether to perform hyperparameter tuning',
                        default=False)
    parser.add_argument('--name',  '-n',
                        dest="name",
                        metavar='NAME',
                        help =  'Name of the model',
                        default='SAE')

    args = parser.parse_args()
    config = config_switch(args.name)(args.tune)
    tune.run()
    train_model(config, tune=args.tune)