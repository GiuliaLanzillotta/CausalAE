# Code for SAE (no attention in the encoder)
# Paper: https://arxiv.org/abs/2006.07796
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE
from torch.nn import functional as F
from .utils import act_switch

class SAE(nn.Module, GenerativeAE):

    def __init__(self, params:dict, dim_in) -> None:
        super(SAE, self).__init__()
        self.latent_size = params["latent_size"]
        self.unit_dim = params["unit_dim"]
        self.N = params["latent_vecs"] # number of latent vectors to store for hybrid sampling
        self.dim_in = dim_in # C, H, W
        self.mode="auto"
        # Building encoder

        conv_net = ConvNet(dim_in, self.latent_size, depth=params["enc_depth"], **params)
        self.conv_net = conv_net # returns vector of latent_dim size
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_size, self.unit_dim, self.N)
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = (self.latent_size, 1, 1)
        self.scm = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)
        self.act = nn.Sigmoid()


    def encode(self, inputs: Tensor):
        codes = self.conv_net(inputs)
        return codes

    def sample_noise(self, codes:Tensor):
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    def sample_noise_from_prior(self, num_samples:int):
        return self.hybrid_layer.sample_from_prior((num_samples,))

    def sample_noise_from_posterior(self, inputs: Tensor):
        codes = self.encode(inputs)
        return self.sample_noise(codes)

    def decode(self, noise:Tensor, activate:bool):
        if self.mode=="auto": x = noise
        else: x = torch.ones(size = (noise.shape[0],)+self.decoder_initial_shape).to(noise.device)
        output = self.scm(x, noise, mode=self.mode)
        if activate: output = self.act(output)
        return output

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate)

    def forward(self, inputs: Tensor, activate:bool=False) -> Tensor:
        inputs = inputs.view((-1, )+self.dim_in)
        codes = self.encode(inputs)
        # normal autoencoder mode (no noise)
        if self.mode=="auto": noise = codes.view((-1,)+self.decoder_initial_shape) #TODO: not working: 2D instead of 3D output
        elif self.mode=="hybrid": noise = self.sample_noise(codes)
        else: raise NotImplementedError
        output = self.decode(noise, activate)
        return  output

    def loss_function(self, *args):
        X_hat = args[0]
        X = args[1]
        MSE = F.mse_loss(self.act(X_hat), X, reduction="mean")
        BCE = F.binary_cross_entropy_with_logits(X_hat, X, reduction="mean")
        return BCE, MSE