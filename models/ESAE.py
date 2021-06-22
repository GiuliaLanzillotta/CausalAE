#Code for E(quivariant)SAE experiment
#Experiment number one on SAE independence
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE, SAE
from torch.nn import functional as F
from .utils import act_switch

class ESAE(SAE):
    """Equivariant version of the SAE"""

    def sample_noise_from_prior(self, num_samples:int):
        pass

    def sample_noise_from_posterior(self, inputs: Tensor):
        pass

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        pass

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False) -> Tensor:
        pass

    def loss_function(self, *args):
        pass
