from .layers import *
from .BASE import GenerativeAE
from .VAE import VAE
from .SAE import SAE
from .ESAE import ESAE

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE}
