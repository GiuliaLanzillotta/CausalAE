from ray import tune
import yaml


def standard_tuning(config):
    config["opt_params"]["LR"] = tune.loguniform(1e-4, 1e-1), #todo: look into it
    config["data_params"]["batch_size"] = tune.randint(100,300)
    # training same model with 8 different manual seeds by default
    config["logging_params"]["manual_seed"] = tune.grid_search([1265, 1234, 5432, 2021, 6732, 11, 29, 50])
    def name_creator_fun(config):
        info = [config['logging_params']['name'],
                config['logging_params']['version'],
                str(config['opt_params']['LR']),
                str(config["data_params"]["batch_size"]),
                str(config["logging_params"]["manual_seed"])]
        return str.join('_', info)
    return config, name_creator_fun

def get_VAE_configs(tuning:bool=False):
    #loading standard configs dictionary
    with open('configs/VAE.yaml', 'r') as file:
        try: config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    if tuning:
        #changing some of the keys
        # standard tuning (regarding optimisation
        config, name_creator_fun = standard_tuning(config)
        config["logging_params"]["name_creator"] = name_creator_fun

    return config

def get_SAE_configs(tuning:bool=False):
    #loading standard configs dictionary
    with open('configs/SAE.yaml', 'r') as file:
        try: config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
            return None
    if tuning:
        #changing some of the keys
        # standard tuning (regarding optimisation
        config, name_creator_fun = standard_tuning(config)
        config["model_params"]["unit_dim"] = tune.grid_search([1, 2, 4])
        config["opt_params"]["auto_epochs"] = tune.grid_search([-1, 10])
        def SAE_name_creator_fun(config):
            name = name_creator_fun(config)
            name+=str(config["model_params"]["unit_dim"])+"_"\
                  +str(config["opt_params"]["auto_epochs"])
            return name
        config["logging_params"]["name_creator"] = SAE_name_creator_fun

    return config

