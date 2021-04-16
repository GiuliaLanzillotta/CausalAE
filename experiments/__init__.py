# Module managing experiments
from . import VAEManager
from . import data
from .VAEManager import VAEXperiment
from .SAEManager import SAEXperiment


experiments_switch = {'VAE':VAEXperiment,
                      'BaseSAE':SAEXperiment}