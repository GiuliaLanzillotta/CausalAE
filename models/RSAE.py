"""Script for the 'Regularised SAE' model"""
from torch import nn
from torch import Tensor
import torch
from torchvision.transforms import Normalize
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE, SAE, utils, VecSCM, VecSCMDecoder, HybridAE
from torch.nn import functional as F
from .utils import act_switch
from abc import ABCMeta, abstractmethod, ABC


class RHybridAE(HybridAE, ABC):
    """Regularised version of the HybridSAE: simply adds a regularisation term
    to the reconstruction objective."""
    @abstractmethod
    def __init__(self, params:dict):
        HybridAE.__init__(self, params)

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False) -> list:
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes)
        output = self.decode(codes, activate)
        return  [output, codes]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        return self.forward(x, activate, update_prior=True)[0]

    def loss_function(self, *args, **kwargs):
        X_hat = args[0]
        Z = args[1]
        X = kwargs["X"]
        lamda = kwargs.get('lamda')
        device = kwargs.get('device','cpu')
        use_MSE = kwargs.get('use_MSE',True)

        if use_MSE:
            # mean over batch of the sum over all other dimensions
            L_rec = torch.sum(F.mse_loss(self.act(X_hat), X, reduction="none"),
                            tuple(range(X_hat.dim()))[1:]).mean()
        else:
            L_rec = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()

        Z_prior = self.hybrid_layer.sample_from_prior(Z.shape)
        L_reg = utils.compute_MMD(Z,Z_prior, kernel=self.kernel_type, **self.params, device=device)
        loss = L_rec + lamda*L_reg
        return{'loss': loss, 'Reconstruction_loss':L_rec, 'Regularization_loss':L_reg}


class RSAE(RHybridAE, SAE):
    """Regularised version of the SAE"""
    def __init__(self, params:dict, dim_in) -> None:
        SAE.__init__(self, params, dim_in)
        self.kernel_type = params["MMD_kernel"]

    def decode(self, noise:Tensor, activate:bool):
        return SAE.decode(self, noise, activate)

class VecRSAE(RHybridAE):
    """Version of RSAE model for vector based (not image based) data"""
    def __init__(self, params: dict, dim_in: int, full: bool) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super(VecRSAE, self).__init__(params)
        self.dim_in = dim_in[0]
        self.kernel_type = params["MMD_kernel"]

        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.full = full
        if not full:
            scm = VecSCM(self.latent_size, self.unit_dim, act=params.get("act"))
            reverse_encoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))
            self.decoder = nn.Sequential(scm, reverse_encoder)
        else: self.decoder = VecSCMDecoder(self.latent_size, self.unit_dim, list(reversed(layers)), act=params.get("act"))


    def decode(self, noise:Tensor, activate:bool):
        # since x is a constant we're always going to get the same output
        if not self.full:
            output = self.decoder(noise)
        else:
            x = torch.ones_like(noise).to(noise.device)
            output = self.decoder(x, noise)
        if activate: output = self.act(output)
        return output

