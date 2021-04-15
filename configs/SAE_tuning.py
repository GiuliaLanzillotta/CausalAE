from ray import tune
import yaml

#loading standard configs dictionary
with open('configs/SAE.yaml', 'r') as file:
    try: config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)
#changing some of the keys
ld = config["model_params"]["latent_dim"]
config["model_params"]["unit_dim"] = tune.choice([1, 2, 4])
config["opt_params"]["LR"] = tune.loguniform(1e-4, 1e-1), #todo: look into it
config["opt_params"]["auto_epochs"] = tune.choice([-1, 10, 50])
config["data_params"]["batch_size"] = tune.choice([144, 256, 432])
config["logging_params"]["manual_seed"] = tune.choice([1265, 1234, 5432, 2021, 6732, 11, 29, 50])

