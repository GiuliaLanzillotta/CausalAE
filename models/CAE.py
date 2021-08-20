"""Set of CAUSAL autoencoder models, i.e. models trained to be equivariant/invariant to interventions in the latent space."""

from abc import abstractmethod, ABC

import numpy as np
import torch
from torch import Tensor, nn

from metrics import LatentInvarianceEvaluator
from . import ConvNet, SCMDecoder, FCBlock, VecSCM, HybridAE, UpsampledConvNet
from .utils import act_switch


class CausalAE(HybridAE, ABC):
    """Causally trained version of the HybridAE: simply adds a regularisation term
    to the reconstruction objective."""
    @abstractmethod
    def __init__(self, params:dict):
        HybridAE.__init__(self, params)
        self.random_state = np.random.RandomState(params.get("random_seed",11))

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False, integrate=True) -> list:
        codes = self.encode(inputs)
        output = self.decode(codes, activate)
        self.hybrid_layer.update_prior(codes, integrate=integrate)
        return  [output]

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        return self.forward(x, activate, update_prior=True, integrate=True)[0]

    def loss_function(self, *args, **kwargs):
        """ kwargs accepted keys:
            - lamda -> regularisation weight
            - device - SE
            - num_samples -> number of samples from hybrid layer (should be >> than latent codes
            number)
            - use_MSE - SE
        """
        X_hat = args[0]
        X = kwargs["X"]
        lamda = kwargs.get('lamda')
        device = kwargs.get('device','cpu')
        num_samples = kwargs.get('num_samples', 100)
        num_interventions = kwargs.get('num_interventions', 20)
        use_MSE = kwargs.get('use_MSE',True)

        MSE,BCE = self.pixel_losses(X,X_hat)
        L_rec = MSE if use_MSE else BCE

        latent_samples = self.hybrid_layer.sample_from_prior((num_samples,-1))
        with torch.no_grad(): responses = self.encode(self.decode(latent_samples.to(device), activate=True))

        #TODO: parallelise to make faster

        invariance_sum = 0.

        for d in range(self.latent_size):
            errors = torch.zeros(self.latent_size, dtype=torch.float).to(device)
            hybrid_posterior = LatentInvarianceEvaluator.posterior_distribution(latent_samples, self.random_state, d)
            for i in range(num_interventions):
                latent_samples_prime = LatentInvarianceEvaluator.noise_intervention(latent_samples, d, hard=True, sampling_fun=hybrid_posterior)
                outputs = self.decode(latent_samples_prime.to(device), activate=True)
                with torch.no_grad(): responses_prime = self.encode(outputs)
                error = torch.linalg.norm((responses-responses_prime), ord=2, dim=0)/num_samples # D x 1 #FIXME: multi-dimensional units to be considered
                errors += (error/error.max()) # D x 1
                del latent_samples_prime
                del responses_prime
            # sum all the errors on non intervened-on dimensions
            invariance_sum += (torch.sum(errors[:d]) + torch.sum(errors[d+1:]))/(num_interventions*self.latent_size) # averaging

        L_reg = invariance_sum
        loss = L_rec + lamda*L_reg
        return{'loss': loss, 'Reconstruction_loss':L_rec, 'Regularization_loss':L_reg}




class XCSAE(CausalAE):

    def __init__(self, params: dict, dim_in) -> None:
        CausalAE.__init__(self, params)
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params)
        fc_net = FCBlock(conv_net.final_dim, [256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, fc_net) # returns vector of latent_dim size
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = conv_net.final_shape
        # 1. vecSCM N -> Z (causal block)
        # - mapping the latent code to the new causal space with an SCM-like structure
        self.caual_block = VecSCM(self.latent_size, **params)
        # 2. SCM Z + constant -> X (decoder)
        # - keeping the SCM like structure in the decoder
        self.decoder = SCMDecoder(self.decoder_initial_shape, dim_in, depth=params["dec_depth"],**params)

    def decode(self, noise:Tensor, activate:bool):
        z = self.caual_block(noise)
        # feeding a constant signal into the decoder
        # the output will be built on top of this constant trough the StrTrf layers
        x = torch.ones((noise.shape[0],)+self.decoder_initial_shape).to(noise.device) # batch x latent
        output = self.decoder(x, z)
        if activate: output = self.act(output)
        return output


class XCAE(CausalAE):

    def __init__(self, params: dict, dim_in) -> None:
        CausalAE.__init__(self, params)
        self.dim_in = dim_in # C, H, W
        # Building encoder
        conv_net = ConvNet(dim_in, depth=params["enc_depth"], **params)
        conv_fin = FCBlock(conv_net.final_dim, [256, 128, self.latent_size], act_switch(params.get("act")))
        self.encoder = nn.Sequential(conv_net, conv_fin) # returns vector of latent_dim size
        # initialise constant image to be used in decoding (it's going to be an image full of zeros)
        self.decoder_initial_shape = conv_net.final_shape
        # 1. vecSCM N -> Z (causal block)
        # - mapping the latent code to the new causal space with an SCM-like structure
        self.caual_block = VecSCM(self.latent_size, **params)
        # 2. SCM Z + constant -> X (decoder)
        dec_init = FCBlock(self.latent_size, [128, 256, conv_net.final_dim], act_switch(params.get("act")))
        deconv_net = UpsampledConvNet(self.decoder_initial_shape, self.dim_in, depth=params["dec_depth"], **params)
        self.decoder = nn.ModuleList([dec_init, deconv_net])

    def decode(self, noise:Tensor, activate:bool):
        z = self.caual_block(noise)
        z_init = self.decoder[0](z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        x = self.decoder[1](z_init)
        if activate: x = self.act(x)
        return x


