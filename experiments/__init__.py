# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, cyclic_beta_schedule, linear_determ_warmup, VAEVecEXperiment
from .SAEManager import AEExperiment, SAEVecExperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment
from .RManager import RegExperiment, RegVecExperiment
from .ModelsManager import GenerativeAEExperiment


def pick_model_manager(model_name):
    if 'VAE' in model_name: return VAEXperiment
    return GenerativeAEExperiment