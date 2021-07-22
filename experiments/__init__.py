# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, cyclic_beta_schedule, linear_determ_warmup, VAEVecEXperiment
from .SAEManager import SAEXperiment, SAEVecExperiment, ConvAEXperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment
from .RSAEManager import RSAEXperiment, RegVecExperiment,RAEXperiment


experiments_switch = {'BetaVAE':VAEXperiment,
                      'BaseSAE':SAEXperiment,
                      'ESAE': ESAEXperiment,
                      'RSAE': RSAEXperiment,
                      'RAE': RAEXperiment,
                      'AE': ConvAEXperiment,
                      'VecVAE':VAEVecEXperiment,
                      'VecSAE':SAEVecExperiment,
                      'VecESAE':ESAEVecExperiment,
                      'VecRSAE':RegVecExperiment,
                      'VecRAE':RegVecExperiment,
                      'VecAE': SAEVecExperiment}
