""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import pytorch_lightning.utilities.seed
import yaml
import argparse
import numpy as np
import os
import glob
import torch
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from experiments import experiments_switch
from configs import config_switch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune



def train_model(config:dict, tuning:bool=False):
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
    if tuning: callbacks.append(TuneReportCallback(metrics, on="validation_end"))

    # resuming from checkpoint
    checkpoint_path = os.path.join(config['logging_params']['save_dir'],
                                   config['logging_params']['name'],
                                   config['logging_params']['version'],
                                   "checkpoints/")
    try:
        latest_checkpoint = max(glob.glob(checkpoint_path + "\*ckpt"), key=os.path.getctime)
    except ValueError:
        #no checpoints
        latest_checkpoint = "null"

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

def do_tuning(config:dict):
    # using Ray tuning functionality to do hyperparameter search
    # see here for details: https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html#using-population-based-training-to-find-the-best-parameters
    #todo: select a search algorithm
    scheduler = ASHAScheduler(
        max_t=config['trainer_params']['max_epochs'],
        grace_period=10, # wait at least 10 epochs
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss", "training_iteration"]) #todo: check what we want to save in final table
    path = os.path.join(config['logging_params']['save_dir'],
                        config['logging_params']['name'],"tuner")
    analysis = tune.run(
        tune.with_parameters(train_model, tuning=True),
        resources_per_trial={
            "cpu": config['tuner_params']['cpus'],
            "gpu": config['trainer_params']['gpus']},
        metric="loss",
        mode="min",
        config=config,
        local_dir=path,
        num_samples=config['tuner_params']['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter)
    analysis.results_df.to_csv(os.path.join(path,"tune_results.csv"))



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAE experiments runner')
    parser.add_argument('--tuning',  '-t',
                        dest="tuning",
                        metavar='TUNE',
                        type=bool,
                        help =  'whether to perform hyperparameter tuning',
                        default=False)
    parser.add_argument('--name',  '-n',
                        dest="name",
                        metavar='NAME',
                        help =  'Name of the model',
                        default='BaseSAE')

    args = parser.parse_args()
    config = config_switch[args.name](args.tuning)
    if args.tuning: do_tuning(config)
    else: train_model(config)