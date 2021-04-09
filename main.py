""" Main module. Reads config file, creates the model and dataloader through experiments managers
and starts training"""
import yaml
import argparse
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
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

experiment = experiments_switch[config['model_params']['name']](config)
# saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=f"{tb_logger.save_dir}",
    filename=f"{config['logging_params']['name']+'_'+config['data_params']['dataset_name']+'_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')}",
    save_top_k=3,
    mode='min',
)
runner = Trainer(default_root_dir=f"{tb_logger.save_dir}",
                 min_epochs=1,
                 logger=tb_logger,
                 log_every_n_steps =100,
                 callbacks=[checkpoint_callback],
                 progress_bar_refresh_rate=20,
                 checkpoint_callback=True,
                 benchmark=False,
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)