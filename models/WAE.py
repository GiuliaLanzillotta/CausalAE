"""Implementation of WAE-MMD model"""

import torch
from torch import Tensor
# Code for beta-VAE
# Paper: https://arxiv.org/pdf/1711.01558.pdf
from torch import nn

from . import utils, ConvAE, Xnet


class WAE(ConvAE, nn.Module):
    """MMD implementation of WAE network with deterministic encoder and additional hybrid layer"""

    def __init__(self, params: dict) -> None:
        """ Accepted keywords
        - all keywords taken in ConvAE
        - prior type \in ['Gaussian','Uniform']
        - prior_scale (float) := sd for the Gaussian and 1/4 extension for Uniform
        - kernel type (str) \in {'RBF' 'IMQ' (standard)}
        """
        super(WAE, self).__init__(params)

        self.kernel_type = params.get("MMD_kernel", "IMQ")
        prior_type = params.get('prior_type', 'Gaussian')
        prior_scale = params.get('prior_scale', 1.0); self.prior_scale = prior_scale
        if prior_type == 'Gaussian':
            self.priorD = torch.distributions.MultivariateNormal(torch.zeros(self.latent_size, requires_grad=False),
                                                        prior_scale*torch.eye(self.latent_size, requires_grad=False))
        elif prior_type == 'Uniform':
            self.priorD = torch.distributions.Uniform(low=-2*prior_scale*torch.ones(self.latent_size, requires_grad=False),
                                                      high=2*prior_scale*torch.ones(self.latent_size,requires_grad=False)) # symmetric


    def forward(self, inputs: Tensor, **kwargs) -> list:
        activate = kwargs.get('activate',False)
        update_prior = kwargs.get('update_prior',False)
        integrate = kwargs.get('integrate',True)
        codes = self.encode(inputs)
        if update_prior: self.hybrid_layer.update_prior(codes, integrate=integrate)
        output = self.decode(codes, activate)
        return  [output, codes]

    def sample_noise_from_prior(self, num_samples: int, prior_mode='self', **kwargs):
        """Equivalent to total hybridisation: every point is hybridised at the maximum level
        3 modes supported: self/posterior/hybrid - self uses the prior distribution defined on the latent space"""
        device= kwargs['device']

        with torch.no_grad():
            if prior_mode=='posterior':
                # shuffling and selecting
                idx = torch.randperm(self.hybrid_layer.prior.shape[0], device=device) #shuffling indices
                return torch.index_select(self.hybrid_layer.prior, 0, idx[:num_samples]).detach()
            if prior_mode=='hybrid': return self.hybrid_layer.sample_from_prior((num_samples,)).to(device)
            if prior_mode=='self':
                return self.priorD.sample([num_samples]).to(device)
        raise NotImplementedError('Requested sampling mode not implemented')

    def get_prior_range(self):
        """ returns a range in format [(min, max)] for every dimension that should contain
        most of the data density (905)"""
        ranges = [(-2.*self.prior_scale, 2.*self.prior_scale) for i in range(self.latent_size)]
        return ranges

    def latent_regularization(self, *args, **kwargs):
        """Computes regularization term to be added to the overall loss of the model.
        Regularization consists in MMD between prior and aggregate posterior"""
        device = kwargs.get('device')
        Z = args[1]
        Z_prior = self.sample_noise_from_prior(num_samples=Z.shape[0], prior_mode='self', device=device).detach()
        reg = utils.compute_MMD(Z, Z_prior, kernel=self.kernel_type, device=device, hierarchy=False,
                                strict=False, standardise=False)
        return reg.to(device)

    def add_regularisation_terms(self, *args, **kwargs):
        lamda = kwargs.get('MMD_lamda',100.0)
        losses = kwargs.get('losses')
        MMD = self.latent_regularization(*args, **kwargs)
        losses['MMD'] = MMD
        losses['loss'] += lamda*MMD
        return losses


class XWAE(WAE, Xnet):

    def __init__(self, params: dict) -> None:
        super(XWAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool):
        z = self.causal_block(noise, self.tau)
        z_init = self.decoder[0](z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        x = self.decoder[1](z_init)
        if activate: x = self.act(x)
        return x

    def decode_from_X(self, x, *args, **kwargs):
        activate = kwargs.get('activate',False)
        z_init = self.decoder[0](x).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        x = self.decoder[1](z_init)
        if activate: x = self.act(x)
        return x

    def add_regularisation_terms(self, *args, **kwargs):
        losses = WAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = Xnet.add_regularisation_terms(self, *args, **kwargs)
        return losses

