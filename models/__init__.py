from .layers import *
from .BASE import GenerativeAE
from .VAE import VAE, VecVAE, VAEBase
from .SAE import SAE, VecSAE,HybridAE
from .ESAE import ESAE, VecESAE, EHybridAE

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE,
                 "ESAE":ESAE,
                 "VecVAE":VecVAE,
                 "VecSAE":VecSAE,
                 "VecESAE":VecESAE}
