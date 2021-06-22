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
        # Building encoder

        conv_net = ConvNet(dim_in, 512, depth=params["enc_depth"], **params)
        fc_net = FCBlock(512, [512, 256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, fc_net) # returns vector of latent_dim size
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_size, self.unit_dim, self.N)
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = (128, 2, 2)
        self.dec_init = FCBlock(self.latent_size, [128, 256, 512, 512], act_switch(params.get("act")))
        self.scm = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)
        self.act = nn.Sigmoid()


    def encode(self, inputs: Tensor):
        codes = self.encoder(inputs)
        return codes

    def encode_mu(self, inputs:Tensor) -> Tensor:
        """ returns latent code (not noise) for given input"""
        return self.encode(inputs)

    def sample_noise_from_prior(self, num_samples:int):
        """Equivalent to total hybridisation: every point is hybridised at the maximum level"""
        return self.hybrid_layer.sample_from_prior((num_samples,))

    def sample_noise_from_posterior(self, inputs: Tensor):
        codes = self.encode(inputs)
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    def decode(self, noise:Tensor, activate:bool):
        # feeding a constant signal into the decoder
        # the output will be built on top of this constant trough the StrTrf layers
        x = torch.ones(size = noise.shape).to(noise.device) # batch x latent
        # passing x through the linear layers does not make much sense:
        # since x is a constant we're always going to get the same output
        x = self.dec_init(x).view((-1, )+self.decoder_initial_shape) # batch x 512
        output = self.scm(x, noise)
        if activate: output = self.act(output)
        return output

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate, update_prior=True)

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False) -> Tensor:
        inputs = inputs.view((-1, )+self.dim_in) # just making sure
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes)
        output = self.decode(codes, activate)
        return  output

    def loss_function(self, *args):
        X_hat = args[0]
        X = args[1]
        # mean over batch of the sum over all other dimensions
        MSE = torch.sum(F.mse_loss(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        BCE = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        return BCE, MSE

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        return self.hybrid_layer.prior_range
