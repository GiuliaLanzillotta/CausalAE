from .layers import *
from .BASE import GenerativeAE
from .VAE import VAE
from .SAE import SAE

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE}
