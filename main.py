""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import json

import pytorch_lightning.utilities.seed
import yaml
import argparse
import numpy as np
import os
import glob
import torch
from pytorch_lightning.profiler import PyTorchProfiler
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from experiments import pick_model_manager
from experiments.EvaluationManager import VectorModelHandler, VisualModelHandler, ModelHandler
from configs import get_config
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray import tune
import ray

from pathlib import Path

from experiments.utils import DAG_pretraining_Callback


def train_model(config:dict, tuning:bool=False, test:bool=False, debugging=False, score=True):
    """ Wrapper for the model training loop to be used by the hyper-parameter tuner"""
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'],
                                  version=config['logging_params']['version'],
                                  log_graph=False)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    pytorch_lightning.utilities.seed.seed_everything(config['logging_params']['manual_seed'])

    # resuming from checkpoint
    base_path = Path(config['logging_params']['save_dir']) / config['logging_params']['name']\
                / config['logging_params']['version']
    checkpoint_path =  base_path / "checkpoints/"
    try:
        latest_checkpoint = max(glob.glob(str(checkpoint_path) + "/*ckpt"), key=os.path.getctime)
        print("Loading latest checkpoint.")
    except ValueError:
        print(f"No checkpoint available at {checkpoint_path}. Training from scratch.")
        latest_checkpoint = None # no checkpoints to restore available

    # callbacks
    callbacks = [ModelCheckpoint(monitor='val_loss', mode="min", dirpath=checkpoint_path)]
    metrics = {"loss":"ptl/val_loss"}
    #after each validation epoch we report the above metric to Ray Tune
    if tuning: callbacks.append(TuneReportCallback(metrics, on="validation_end"))
    if config['logging_params']['name']=="CausalVAE" and False: #if the model is CausalVAE we add the DAG learning pretraining step
        callbacks.append(DAG_pretraining_Callback(config["opt_params"]["pretraining"]))

    #save hyperparameters
    hparams_path = base_path / "configs.yaml"
    os.makedirs(base_path, exist_ok=True)
    if not os.path.exists(hparams_path):
        with open(hparams_path, 'w') as out:
            yaml.dump(config, out, default_flow_style=False)


    if debugging:
        # see here: https://pytorch-lightning.readthedocs.io/en/latest/advanced/profiler.html
        profiler = PyTorchProfiler(on_trace_ready=torch.profiler.tensorboard_trace_handler(base_path),
                                   profile_memory=True)

    runner = Trainer(logger=tb_logger,
                     callbacks=callbacks,
                     weights_summary=None,
                     resume_from_checkpoint=latest_checkpoint,
                     reload_dataloaders_every_epoch=config['data_params']['reload_dataloaders_every_epoch'],
                     profiler=profiler if debugging else None,
                     #plugins=DDPPlugin(find_unused_parameters=False) if config['trainer_params']['accelerator']=='ddp' else None,
                     **config['trainer_params'])


    experiment = pick_model_manager(config['model_params']['name'])(config)

    if not test:
        print(f"======= Training {config['model_params']['name']} =======")
        runner.fit(experiment)
        print("Training completed.")
        print("Saving final checkpoint")
        runner.save_checkpoint(str(checkpoint_path/"final.ckpt"))
    else:
        print(f"======= Testing {config['model_params']['name']} =======")
        runner.test(experiment)
        print("Testing finished. ")

    if score:
        print("Scoring the model")
        handler = ModelHandler.from_experiment(experiment)
        handler.score_model(save_scores=True,
                            update_general_scores=True,
                            response_classification=True,
                            random_seed=config["logging_params"]["manual_seed"],
                            inference=config['data_params']['dataset_name']!='3DS',
                            **config['eval_params'])
        if config['eval_params'].get("latent_responses",True):
            handler.latent_responses(**config['eval_params'], store=True)

def do_tuning(config:dict):
    # using Ray tuning functionality to do hyperparameter search
    # see here for details: https://docs.ray.io/en/master/tune/tutorials/tune-pytorch-lightning.html#using-population-based-training-to-find-the-best-parameters
    #todo: select a search algorithm
    scheduler = ASHAScheduler(
        time_attr = "training_step",
        max_t=config['trainer_params']['max_steps'],
        grace_period=10, # wait at least 10 epochs
        reduction_factor=2)
    reporter = CLIReporter(metric_columns=["loss", "training_iteration"]) #todo: check what we want to save in final table
    path = Path('.') / config['logging_params']['save_dir'] / config['logging_params']['name'] / "tuner"
    analysis = tune.run(
        tune.with_parameters(train_model, tuning=True),
        metric="loss",
        mode="min",
        config=config,
        local_dir=path,
        #num_samples=config['tuner_params']['num_samples'],
        scheduler=scheduler,
        progress_reporter=reporter)
    analysis.results_df.to_csv( path / "tune_results.csv")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SAE experiments runner')
    parser.add_argument('--tuning',  '-t',
                        dest="tuning",
                        metavar='TUNE',
                        type=bool,
                        help ='whether to perform hyperparameter tuning',
                        default=False)
    parser.add_argument('--name',  '-n',
                        dest="name",
                        metavar='NAME',
                        help =  'Name of the model',
                        default='CausalVAE')
    parser.add_argument('--data', '-d',
                        dest="data",
                        metavar="DATA",
                        help = 'Name of the dataset to use',
                        default="Pendulum")
    parser.add_argument('--version', '-v',
                        dest="version",
                        metavar="VERSION",
                        help= "Name of version to use",
                        default="dummy")
    parser.add_argument('--data_version', '-dv',
                        dest="data_version",
                        metavar="DATA_VERSION",
                        help= "Name of data version to use (available only for synthetic datasets for now)",
                        default="continuous")
    parser.add_argument('--test', '-e', #Note that when testing is switched on then training is switched off
                        dest="test",
                        metavar="TEST",
                        help= "Whether to load the model for testing",
                        default=False)
    parser.add_argument('--debugging', '-db', #Note that when testing is switched on then training is switched off
                        dest="debug",
                        metavar="DEBUG",
                        help= "Whether to enter debugging mode (pytorch profiler on)",
                        default=False)
    parser.add_argument('--score', '-s', #Note that when testing is switched on then training is switched off
                        dest="score",
                        metavar="SCORE",
                        help= "Whether to score the model at the end of training/testing",
                        default=True)


    args = parser.parse_args()
    config = get_config(args.tuning, args.name, args.data, args.version, args.data_version)
    if args.tuning: do_tuning(config)
    else: train_model(config, test=args.test, debugging=args.debug, score=args.score)