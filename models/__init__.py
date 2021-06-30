from .layers import *
from .BASE import GenerativeAE
from .VAE import VAE, VecVAE
from .SAE import SAE, VecSAE
from .ESAE import ESAE, VecESAE

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE,
                 "ESAE":ESAE,
                 "VecBetaVAE":VecVAE,
                 "VecSAE":VecSAE,
                 "VecESAE":VecESAE}
