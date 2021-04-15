""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import pytorch_lightning.utilities.seed
import yaml
import argparse
import numpy as np
import torch
from experiments import experiments_switch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ray.tune.integration.pytorch_lightning import TuneReportCallback



def train_model(config:dict, num_epochs=10, num_gpus=0):
    """ Wrapper for the model training loop to be used by the hyper-parameter tuner"""
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['logging_params']['name'],
                                  version=config['logging_params']['version'],
                                  log_graph=False)

    # For reproducibility
    torch.manual_seed(config['logging_params']['manual_seed'])
    np.random.seed(config['logging_params']['manual_seed'])
    pytorch_lightning.utilities.seed.seed_everything(config['logging_params']['manual_seed'])

    experiment = experiments_switch[config['model_params']['name']](config)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss')
    metrics = {"loss":"ptl/val_loss"}
    #after each validation epoch we report the above metric to Ray Tune
    tune_report_callback = TuneReportCallback(metrics, on="validation_end")
    # optional: resume from checkpoint... TODO
    runner = Trainer(min_epochs=1,
                     logger=tb_logger,
                     log_every_n_steps=50,
                     callbacks=[checkpoint_callback, tune_report_callback],
                     progress_bar_refresh_rate=20,
                     checkpoint_callback=True,
                     benchmark=False,
                     deterministic=True,
                     **config['trainer_params'])

    print(f"======= Training {config['model_params']['name']} =======")
    runner.fit(experiment)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAE experiments runner')
    parser.add_argument('--tune',  '-c', #TODO
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/SAE.yaml')

    parser.add_argument('--config',  '-c',
                        dest="filename",
                        metavar='FILE',
                        help =  'path to the config file',
                        default='configs/SAE.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try: config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)