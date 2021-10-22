# Module managing experiments
from . import data
from .data import DatasetLoader
from .utils import cyclic_beta_schedule, linear_determ_warmup, get_causal_block_graph
from .ModelsManager import GenerativeAEExperiment


def pick_model_manager(model_name):
    return GenerativeAEExperiment