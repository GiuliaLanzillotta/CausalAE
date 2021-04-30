# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment
from .SAEManager import SAEXperiment


experiments_switch = {'BetaVAE':VAEXperiment,
                      'BaseSAE':SAEXperiment}