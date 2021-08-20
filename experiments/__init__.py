# Module managing experiments
from . import VAEManager
from . import data
from .data import DatasetLoader
from .VAEManager import VAEXperiment, cyclic_beta_schedule, linear_determ_warmup, VAEVecEXperiment
from .SAEManager import AEExperiment, SAEVecExperiment
from .ESAEManager import ESAEXperiment, ESAEVecExperiment
from .RManager import RegExperiment, RegVecExperiment


experiments_switch = {'BetaVAE':VAEXperiment,
                      'ESAE': ESAEXperiment,
                      'AE': AEExperiment,
                      'BaseSAE':AEExperiment,
                      'XAE': AEExperiment,
                      'XSAE': AEExperiment,
                      'RSAE': RegExperiment,
                      'RAE': RegExperiment,
                      'XCAE': RegExperiment,
                      'XCSAE': RegExperiment,
                      'VecVAE':VAEVecEXperiment,
                      'VecSAE':SAEVecExperiment,
                      'VecESAE':ESAEVecExperiment,
                      'VecRSAE':RegVecExperiment,
                      'VecRAE':RegVecExperiment,
                      'VecAE': SAEVecExperiment}
