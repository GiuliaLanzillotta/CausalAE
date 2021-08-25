from .layers import *
from .BASE import GenerativeAE, HybridAE, Xnet
from .VAE import VAE, VecVAE, VAEBase, XVAE
from .SAE import SAE, VecSAE, XSAE
from .ESAE import ESAE, VecESAE, EHybridAE
from .AE import ConvAE, VecAE, XAE
from .RSAE import RSAE, VecRSAE, RHybridAE, RAE, VecRAE
from .CAE import CausalAE, XCSAE, XCAE, XCVAE
from .utils import KL_multiple_univariate_gaussians

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE,
                 "ESAE":ESAE,
                 "RSAE":RSAE,
                 "RAE":RAE,
                 "AE": ConvAE,
                 "XAE": XAE,
                 "XSAE": XSAE,
                 "XVAE": XVAE,
                 "XCAE": XCAE,
                 "XCSAE": XCSAE,
                 "XCVAE": XCVAE,
                 "VecVAE":VecVAE,
                 "VecSAE":VecSAE,
                 "VecESAE":VecESAE,
                 "VecRSAE":VecRSAE,
                 "VecRAE":VecRAE,
                 "VecAE": VecAE}
