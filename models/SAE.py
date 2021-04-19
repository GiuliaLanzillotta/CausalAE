# Code for SAE (no attention in the encoder)
# Paper: https://arxiv.org/abs/2006.07796
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock
from torch.nn import functional as F

class SAE(nn.Module):
    def __init__(self, params:dict, dim_in) -> None:
        super(SAE, self).__init__()
        self.latent_size = params["latent_size"]
        self.unit_dim = params["unit_dim"]
        self.N = params["latent_vecs"] # number of latent vectors to store for hybrid sampling
        self.dim_in = dim_in # C, H, W
        # Building encoder
        #TODO: add a selection for non-linearity here

        conv_net = ConvNet(dim_in, 256, depth=params["enc_depth"], **params)
        self.conv_net = conv_net # returns vector of latent_dim size
        fc_class = FCResidualBlock if params["residual_fc"] else FCBlock
        self.fc = fc_class(256, [128, 64,  self.latent_size], nn.ReLU)
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_size, self.unit_dim, self.N)
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = (self.latent_size, 1, 1)
        self.scm = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)
        self.act = nn.Sigmoid()

    def encode(self, inputs: Tensor):
        conv_net_out = self.conv_net(inputs)
        codes = self.fc(conv_net_out)
        return codes

    def sample_noise(self, codes:Tensor):
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    def decode(self, z: Tensor, x:Tensor, mode:str) -> Tensor:
        output = self.scm(x, z, mode=mode)
        return self.act(output)

    def generate_from_prior(self, num_samples:int):
        """Samples (with hybrid sampling) from the latent space."""
        noise = self.hybrid_layer.sample_from_prior(num_samples)
        x = torch.ones(size = (num_samples,)+self.decoder_initial_shape)
        return self.decode(noise, x, mode="hybrid")

    def generate(self, x: Tensor) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x)

    def forward(self, inputs: Tensor, mode="auto") -> Tensor:
        codes = self.encode(inputs)
        if mode=="auto":
            # normal autoencoder mode (no noise)
            x = codes.view((-1,)+self.decoder_initial_shape)
            z = x
        elif mode=="hybrid":
            with torch.no_grad():
                x = torch.ones(size = (codes.shape[0],)+self.decoder_initial_shape).to(codes.device)
            z = self.sample_noise(codes)
        output = self.decode(z, x, mode=mode)
        return  output

    @staticmethod
    def loss_function(*args):
        #TODO: include FID
        X_hat = args[0]
        X = args[1]
        FID = 0
        MSE = F.mse_loss(X_hat, X, reduction="sum")
        BCE = F.binary_cross_entropy(X_hat, X, reduction="sum")
        return BCE, FID, MSE