from .layers import *
from .BASE import GenerativeAE, HybridAE, Xnet
from .VAE import VAE, VecVAE, VAEBase, XVAE
from .AVAE import AVAE
from .SAE import SAE, VecSAE, XSAE
from .ESAE import ESAE, VecESAE, EHybridAE
from .AE import ConvAE, VecAE, XAE
from .RSAE import RSAE, VecRSAE, RHybridAE, RAE, VecRAE
from .CAE import CausalAE, XCSAE, XCAE, XCVAE, XCWAE
from .CausalVAE import CausalVAE
from .WAE import WAE, XWAE
from .INET import INET
from .utils import KL_multiple_univariate_gaussians

models_switch = {"BetaVAE":VAE,
                 "BaseSAE":SAE,
                 "ESAE":ESAE,
                 "RSAE":RSAE,
                 "RAE":RAE,
                 "AE": ConvAE,
                 "WAE": WAE,
                 "AVAE":AVAE,
                 "XAE": XAE,
                 "XSAE": XSAE,
                 "XVAE": XVAE,
                 "XWAE": XWAE,
                 "XCAE": XCAE,
                 "XCSAE": XCSAE,
                 "XCVAE": XCVAE,
                 "XCWAE": XCWAE,
                 "VecVAE":VecVAE,
                 "VecSAE":VecSAE,
                 "VecESAE":VecESAE,
                 "VecRSAE":VecRSAE,
                 "VecRAE":VecRAE,
                 "VecAE": VecAE,
                 "CausalVAE":CausalVAE}
