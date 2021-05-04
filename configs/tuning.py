from ray import tune
import os
import yaml



def standard_tuning(config):
    config["opt_params"]["LR"] = tune.loguniform(1e-4, 1e-1), #todo: look into it
    config["data_params"]["batch_size"] = tune.randint(100,300)
    # training same model with 8 different manual seeds by default
    config["logging_params"]["manual_seed"] = tune.grid_search([1265, 1234, 5432, 2021, 6732, 11, 29, 50])
    return config


def get_VAE_tuning_configs(config:dict):
    config = standard_tuning(config)
    return config

def get_BaseSAE_tuning_configs(config:dict):
    #changing some of the keys
    # standard tuning (regarding optimisation
    config = standard_tuning(config)
    config["model_params"]["unit_dim"] = tune.grid_search([1, 2, 4])
    config["opt_params"]["auto_epochs"] = tune.grid_search([-1, 10])
    return config


config_switch = {'VAE':get_VAE_tuning_configs,
                 'BaseSAE':get_BaseSAE_tuning_configs}


def get_config(tuning:bool, model_name:str, data:str, version:str):
    """ Preparing the config file that will be used during training.
    A unique config file will be assembled from the multiple config files"""
    model_path = str.join("/",["configs","models",model_name,version+".yaml"])
    standard_model_path = str.join("/",["configs","models",model_name,"standard.yaml"])
    data_path = str.join("/", ["configs", "data", data+".yaml"])

    # loading the base config file: this will be updated
    with open('configs/standard.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # load from file if already existing (this is the case when the training has
    # already been launched previously and then interrupted)
    hparams_path = str.join("/", [".",config['logging_params']['save_dir'],model_name, version+"_"+data, "configs.yaml"])
    if os.path.exists(hparams_path):
        print("Found existing config file: loading.")
        with open(hparams_path, 'r') as file:
            config = yaml.safe_load(file)
            if tuning: config = config_switch[model_name](config)
            return config
    # BUILDING CONFIG FILE ---
    # loading the standard model config (each model has a standard version config)
    with open(standard_model_path, 'r') as file:
        standard_fig = yaml.safe_load(file)
        for k in standard_fig.keys():
            config[k].update(standard_fig[k])
    # including data parameters
    with open(data_path, 'r') as file:
        data_fig = yaml.safe_load(file)
        config["data_params"].update(data_fig)
    if version!="standard":
        # updating the hyper-parameters with the ones specific to this version
        # note that the version config file could contain hyper-parameters regarding data figs
        with open(model_path, 'r') as file:
            model_fig = yaml.safe_load(file)
            for k in model_fig.keys():
                config[k].update(model_fig[k])
    # updating logging information
    config["vis_params"]["plot_every"] = data_fig["plot_every"]
    logging_fig = {
        "score_every":data_fig["score_every"],
        "name":model_name,
        "version":version+"_"+data
    }
    config["logging_params"].update(logging_fig)
    # finally tuning
    if tuning: config = config_switch[model_name](config)
    return config

