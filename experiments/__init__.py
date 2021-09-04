# Module managing experiments
from . import data
from .data import DatasetLoader
from .utils import cyclic_beta_schedule, linear_determ_warmup, get_causal_block_graph
from .VAEManager import VAEXperiment, VAEVecEXperiment
from .SAEManager import AEExperiment, SAEVecExperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment
from .RManager import RegExperiment, RegVecExperiment
from .ModelsManager import GenerativeAEExperiment
from . import VAEManager


def pick_model_manager(model_name):
    return GenerativeAEExperiment