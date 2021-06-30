# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, cyclic_beta_schedule, linear_determ_warmup, VAEVecEXperiment
from .SAEManager import SAEXperiment, SAEVecExperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment


experiments_switch = {'BetaVAE':VAEXperiment,
                      'BaseSAE':SAEXperiment,
                      'ESAE': ESAEXperiment,
                      'VecBetaVAE':VAEVecEXperiment,
                      'VecBaseVAE':SAEVecExperiment,
                      'VecESAE':ESAEVecExperiment}