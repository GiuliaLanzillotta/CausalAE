"""Set of CAUSAL autoencoder models, i.e. models trained to be equivariant/invariant to interventions in the latent space."""

from abc import abstractmethod, ABC
from typing import List

import numpy as np
import torch
from torch import Tensor, nn

from metrics import LatentInvarianceEvaluator
from . import HybridAE, GenerativeAE, utils, XSAE, XAE, VAEBase, XVAE, VAE, Xnet, VecSCM
from .utils import KL_multiple_univariate_gaussians


class CausalNet(GenerativeAE, ABC):
    """ Superclass of all causal networks, i.e. networks that admit causal training."""

    def initialise_causal_net(self, random_seed):
        self.random_state = np.random.RandomState(random_seed)

    @abstractmethod
    def compute_errors_from_responses(self, R, R_prime):
        """Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        raise NotImplementedError

    def compute_invariance_loss(self, inputs, **kwargs):
        """ kwargs accepted keys:
            - num_samples -> number of samples for each intervention
            - num_interventions -> number of interventions to use to compute invariance score
            - device - SE
        """
        device = kwargs.get('device','cpu')
        num_samples = kwargs.get('num_samples', 100)
        num_interventions = kwargs.get('num_interventions', 20)

        with torch.no_grad():
            prior_samples = self.sample_noise_from_prior(num_samples, ).to(device)
            #FIXME: aggregate posterior estimate instead of using only one batch updating it
            posterior_samples = self.sample_noise_from_posterior(inputs).to(device)
            # note: the output of the 'encode' method could be whatever (e.g. a list, a Tensor)
            responses = self.encode(self.decode(prior_samples, activate=True))

        #TODO: parallelise to make faster

        invariance_sum = 0.

        for d in range(self.latent_size):
            errors = torch.zeros(self.latent_size, dtype=torch.float).to(device)
            hybrid_posterior = LatentInvarianceEvaluator.posterior_distribution(posterior_samples, self.random_state, d)
            for i in range(num_interventions):
                prior_samples_prime = LatentInvarianceEvaluator.noise_intervention(prior_samples, d, hard=True, sampling_fun=hybrid_posterior)
                outputs = self.decode(prior_samples_prime.to(device), activate=True)
                with torch.no_grad():  #FIXME
                    responses_prime = self.encode(outputs)
                #FIXME: multi-dimensional units to be considered
                error = self.compute_errors_from_responses(responses, responses_prime)
                errors += (error/error.max()) # D x 1
            # sum all the errors on non intervened-on dimensions
            invariance_sum += (torch.sum(errors[:d]) + torch.sum(errors[d+1:]))/(num_interventions*self.latent_size) # averaging

        return invariance_sum

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        losses = kwargs.get('losses')
        lamda = kwargs.get('invariance_lamda')
        X = kwargs["X"]
        L_reg = self.compute_invariance_loss(inputs = X, **kwargs)
        losses['Invariance_loss'] = L_reg
        losses['loss'] += lamda*L_reg
        return losses


class CausalAE(CausalNet, ABC):
    """Causally trained version of the HybridAE: simply adds a regularisation term
    to the reconstruction objective."""

    def compute_errors_from_responses(self, R, R_prime):
        """
        R, R_prime: Tensors of dimension m x D
        Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        return LatentInvarianceEvaluator.compute_absolute_errors(R,R_prime)

class XCSAE(XSAE, CausalAE):

    def __init__(self, params: dict) -> None:
        XSAE.__init__(self, params)
        CausalAE.initialise_causal_net(self, params['random_seed'])

        #CausalAE.__init__(self, params)
        #XSAE.__init__(self, params, dim_in)

    def decode(self, noise:Tensor, activate:bool):
        return XSAE.decode(self, noise, activate)

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalAE.add_regularisation_terms(*args, **kwargs)
        kwargs['losses'] = losses
        losses = XSAE.add_regularisation_terms(*args, **kwargs)
        return losses

class XCAE(XAE, CausalAE):

    def __init__(self, params: dict) -> None:
        XAE.__init__(self, params)

    def decode(self, noise:Tensor, activate:bool):
        return XAE.decode(self, noise, activate)

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalAE.add_regularisation_terms(*args, **kwargs)
        kwargs['losses'] = losses
        losses = XAE.add_regularisation_terms(*args, **kwargs)
        return losses


class CausalVAE(VAE, CausalNet, ABC):
    """ Causally trained version on VAE network - i.e. any net that implements VAEBase"""
    def __init__(self, params:dict):
        VAE.__init__(self, params)
        self.random_state = np.random.RandomState(params["random_seed"])

    def compute_errors_from_responses(self, R:List[Tensor], R_prime:List[Tensor]):
        return LatentInvarianceEvaluator.compute_distributional_errors(R, R_prime)

    def add_regularisation_terms(self, *args, **kwargs):
        losses = VAE.add_regularisation_terms(*args, **kwargs)
        kwargs['losses'] = losses
        losses = CausalNet.add_regularisation_terms(*args, **kwargs)
        return losses

class XCVAE(CausalVAE, Xnet):
    """ XVAE net augmented with causal training """

    def __init__(self, params: dict) -> None:
        CausalVAE.__init__(self, params)
        self.sparsity_on = params.get("sparsity",False)
        self.caual_block = VecSCM(use_masking = self.sparsity_on, **params)

    def decode(self, noise:Tensor, activate:bool):
        Z = self.caual_block(noise)
        out_init = self.decoder[0](Z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](out_init)
        if activate: out = self.act(out)
        return out

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalVAE.add_regularisation_terms(*args, **kwargs)
        kwargs['losses'] = losses
        losses = Xnet.add_regularisation_terms(*args, **kwargs)
        return losses
