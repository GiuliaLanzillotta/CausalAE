# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, cyclic_beta_schedule, linear_determ_warmup
from .SAEManager import SAEXperiment
from .ESAEManager import ESAEXperiment


experiments_switch = {'BetaVAE':VAEXperiment,
                      'BaseSAE':SAEXperiment,
                      'ESAE': ESAEXperiment}