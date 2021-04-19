from .layers import ConvBlock, ConvNet, TransConvBlock, TransConvNet, \
    StrTrf, SCMDecoder, GaussianLayer, HybridLayer, FCBlock
from .VAE import VAE
from .SAE import SAE

models_switch = {'VAE':VAE}
