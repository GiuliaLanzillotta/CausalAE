from .layers import *
from .BASE import GenerativeAE, HybridAE
from .VAE import VAE, VecVAE, VAEBase
from .SAE import SAE, VecSAE
from .ESAE import ESAE, VecESAE, EHybridAE
from .AE import ConvAE, VecAE
from .RSAE import RSAE, VecRSAE, RHybridAE, RAE, VecRAE

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE,
                 "ESAE":ESAE,
                 "RSAE":RSAE,
                 "RAE":RAE,
                 "VecVAE":VecVAE,
                 "VecSAE":VecSAE,
                 "VecESAE":VecESAE,
                 "VecRSAE":VecRSAE,
                 "VecRAE":VecRAE}
