# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, TransConvNet, GaussianLayer, GenerativeAE, UpsampledConvNet, FCBlock, DittadiConvNet, DittadiUpsampledConv, VecSCM, VecSCMDecoder
from torch.nn import functional as F
from .utils import act_switch

class VAE(nn.Module, GenerativeAE):

    def __init__(self, params:dict, dim_in) -> None:
        super(VAE, self).__init__()
        self.beta = params["beta"]
        self.latent_size = params["latent_size"]
        self.dittadi_v = params["dittadi"] # boolean flag determining whether or not to use Dittadi convolutional structure
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, 256, depth=params["enc_depth"], **params) if not self.dittadi_v \
            else DittadiConvNet(self.latent_size)
        self.conv_net = conv_net
        if not self.dittadi_v:
            self.fc_enc = FCBlock(256, [128, 64, self.latent_size], act_switch(params["act"]))
            self.fc_dec = FCBlock(self.latent_size, [64, 128, 256], act_switch(params["act"]))
        self.gaussian_latent = GaussianLayer(self.latent_size, self.latent_size, params["gaussian_init"])
        self.upsmpld_conv_net = UpsampledConvNet((64, 2, 2), self.dim_in, depth=params["dec_depth"], **params) \
            if not self.dittadi_v else DittadiUpsampledConv(self.latent_size)
        self.act = nn.Sigmoid()


    def encode(self, inputs: Tensor):
        conv_result = self.conv_net(inputs)
        if not self.dittadi_v: codes = self.fc_enc(conv_result)
        else: codes = conv_result
        z, logvar, mu = self.gaussian_latent(codes)
        return [z, mu, logvar]

    def encode_mu(self, inputs:Tensor) -> Tensor:
        """ returns latent code (not noise) for given input"""
        return self.encode(inputs)[0]

    def decode(self, noise: Tensor, activate:bool) -> Tensor:
        if not self.dittadi_v: noise = self.fc_dec(noise)
        upsmpld_res = self.upsmpld_conv_net(noise)
        if activate: upsmpld_res = self.act(upsmpld_res)
        return upsmpld_res

    def sample_noise_from_prior(self, num_samples:int):
        return self.gaussian_latent.sample_standard(num_samples)

    def sample_noise_from_posterior(self, inputs: Tensor):
        return self.encode(inputs)[0]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate)[0]

    def forward(self, inputs: Tensor, activate:bool=False) -> list:
        inputs = inputs.view((-1, )+self.dim_in)
        z, mu, logvar = self.encode(inputs)
        return  [self.decode(z, activate), mu, logvar]

    def loss_function(self, *args, **kwargs) -> dict:
        X_hat = args[0]
        mu = args[1]
        log_var = args[2]
        X = kwargs["X"]
        #  In this context it makes sense to normalise β by latent z size
        # m and input x size n in order to compare its different values
        # across different latent layer sizes and different datasets
        KL_weight = kwargs["KL_weight"] # usually weight = M/N
        use_MSE = kwargs.get("use_MSE",True)
        # ELBO = reconstruction term + prior-matching term
        # Note: for both losses we take the average over the batch and sum over the other dimensions
        BCE = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                               tuple(range(X_hat.dim()))[1:]).mean() #sum over all dimensions except the first one (batch)
        MSE = torch.sum(F.mse_loss(self.act(X_hat), X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:]).mean()
        recons_loss = MSE if use_MSE else BCE
        KL_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), 1).mean()
        loss = recons_loss + self.beta * KL_weight * KL_loss
        return {'loss': loss, 'Reconstruction_loss':recons_loss, 'KL':KL_loss, 'MSE':MSE, 'BCE':BCE}

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        return self.gaussian_latent.prior_range


class VecVAE(VAE):
    """Version of VAE model for vector based (not image based) data"""
    def __init__(self, params:dict, dim_in:int, full:bool) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super(VecVAE, self).__init__(params, dim_in)
        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.full = full
        if not full:
            scm = VecSCM(self.latent_size, self.unit_dim, act=params.get("act"))
            reverse_encoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))
            self.decoder = nn.Sequential(scm, reverse_encoder)
        else: self.decoder = VecSCMDecoder(self.latent_size, self.unit_dim, reversed(layers), act=params.get("act"))

    def encode(self, inputs: Tensor):
        codes = self.encoder(inputs)
        z, logvar, mu = self.gaussian_latent(codes)
        return [z, mu, logvar]


    def decode(self, noise:Tensor, activate:bool):
        # since x is a constant we're always going to get the same output
        if not self.full:
            output = self.decoder(noise)
        else:
            x = torch.ones_like(noise).to(noise.device)
            output = self.decoder(x, noise)
        if activate: output = self.act(output)
        return output