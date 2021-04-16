from ray import tune
import yaml


def standard_tuning(config):
    config["opt_params"]["LR"] = tune.loguniform(1e-4, 1e-1), #todo: look into it
    config["data_params"]["batch_size"] = tune.choice([144, 256, 432])
    # training same model with 8 different manual seeds by default
    config["logging_params"]["manual_seed"] = tune.choice([1265, 1234, 5432, 2021, 6732, 11, 29, 50])
    return config

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
        config = standard_tuning(config)

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
        config = standard_tuning(config)
        config["model_params"]["unit_dim"] = tune.choice([1, 2, 4])
        config["opt_params"]["auto_epochs"] = tune.choice([-1, 10, 50])

    return config

