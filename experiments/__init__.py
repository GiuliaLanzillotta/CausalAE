# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, VAEVecEXperiment
from .SAEManager import AEExperiment, SAEVecExperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment
from .RManager import RegExperiment, RegVecExperiment
from .ModelsManager import GenerativeAEExperiment
from .utils import cyclic_beta_schedule, linear_determ_warmup


def pick_model_manager(model_name):
    return GenerativeAEExperiment