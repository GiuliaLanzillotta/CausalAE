# Code for SAE (no attention in the encoder)
# Paper: https://arxiv.org/abs/2006.07796
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer
from torch.nn import functional as F

class SAE(nn.Module):
    def __init__(self, params:dict, dim_in, device) -> None:
        super(SAE, self).__init__()
        self.latent_dim = params["latent_dim"]
        self.unit_dim = params["unit_dim"]
        self.N = params["latent_vecs"] # number of latent vectors to store for hybrid sampling
        self.dim_in = dim_in # C, H, W
        # Building encoder
        #TODO: add a selection for non-linearity here
        channels_list_enc = params["channels_list_enc"]
        channels_list_dec = params["channels_list_dec"]
        conv_net = ConvNet(dim_in, self.latent_dim, channels_list=channels_list_enc)
        self.conv_net = conv_net # returns vector of latent_dim size
        # hybrid sampling to get the noise vector
        self.hybrid_layer = HybridLayer(self.latent_dim, self.unit_dim, self.N)
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = (self.latent_dim, 1, 1)
        self.scm = SCMDecoder(self.decoder_initial_shape, final_shape=dim_in, latent_size=self.latent_dim,
                              unit_dim=params["unit_dim"], channels_list=channels_list_dec, filter_size=params["filter_size"],
                              stride= params["stride"], upsampling_factor=params["upsampling_factor"])
        self.act = nn.Sigmoid()

    def encode(self, inputs: Tensor):
        codes = self.conv_net(inputs)
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
        return self.decode(noise) #TODO: fix here

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
                x = torch.zeros(size = (codes.shape[0],)+self.decoder_initial_shape).to(codes.device)
            z = self.sample_noise(codes)
        output = self.decode(z, x, mode=mode)
        return  output

    def loss_function(self, *args):
        #TODO: include FID
        X_hat = args[0]
        X = args[1]
        FID = 0
        BCE = F.binary_cross_entropy(X_hat, X, reduction="sum")
        return BCE, FID