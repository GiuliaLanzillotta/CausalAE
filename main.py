""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import yaml
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from models import models_switch
from experiments import experiments_switch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


#TODO: implement ladder/hierarchical VAE and finally SAE
#TODO: include new dataset


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/VAE.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try: config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['logging_params']['name'])

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
#TODO: check these 2
cudnn.deterministic = True
cudnn.benchmark = False

experiment = experiments_switch[config['model_params']['name']](config)
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=f"{tb_logger.save_dir}",
    filename=f"{config['logging_params']['name']+'_'+config['data_params']['dataset_name']+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    save_top_k=3,
    mode='min',
)
#TODO: check trainer arguments
runner = Trainer(default_save_path=f"{tb_logger.save_dir}",
                 min_nb_epochs=1,
                 logger=tb_logger,
                 log_save_interval=100,
                 train_percent_check=1.,
                 val_percent_check=1.,
                 num_sanity_val_steps=5,
                 early_stop_callback = False,
                 **config['trainer_params'])
print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)