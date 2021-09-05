"""Set of CAUSAL autoencoder models, i.e. models trained to be equivariant/invariant to interventions in the latent space."""

from abc import abstractmethod, ABC
from typing import List

import copy
import numpy as np
import torch
from torch import Tensor, nn

from metrics import LatentInvarianceEvaluator
from . import HybridAE, GenerativeAE, utils, XSAE, XAE, VAEBase, XVAE, VAE, Xnet, VecSCM
from .utils import KL_multiple_univariate_gaussians


class CausalNet(GenerativeAE, ABC):
    """ Superclass of all causal networks, i.e. networks that admit causal training."""

    def __init__(self, params):
        super().__init__(params)
        self.random_state = np.random.RandomState(params['random_seed'])

    @abstractmethod
    def compute_errors_from_responses(self, R, R_prime, **kwargs):
        """Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        raise NotImplementedError

    def compute_invariance_loss(self, inputs, **kwargs):
        """ kwargs accepted keys:
            - num_samples -> number of samples for each intervention
            - num_interventions -> number of interventions to use to compute invariance score
            - device - SE

        Note that although similar in scope the computation performed here is not equivalent to the
        one used to evaluate the model's noise invariance at evaluation time
            - the posterior distribution is obtained from a single batch here
            - the errors are not normalised
        """
        device = kwargs.get('device','cpu')
        num_samples = kwargs.get('num_samples', 10)
        num_interventions = kwargs.get('num_interventions', 10)
        prior_mode = kwargs.get('prior_mode','posterior')

        with torch.no_grad():
            prior_samples = self.sample_noise_from_prior(num_samples, prior_mode=prior_mode).to(device)
            #Note that we're using only the available batch to approximaate the aggregate posterior estimate
            posterior_samples = self.sample_noise_from_posterior(inputs).to(device)
            # note: the output of the 'encode' method could be whatever (e.g. a list, a Tensor)
            responses = self.encode(self.decode(prior_samples, activate=True)) # n x d


        all_prior_samples = []
        num_units = self.latent_size//self.unit_dim
        for u in range(num_units):
            hybrid_posterior = LatentInvarianceEvaluator.posterior_distribution(posterior_samples,self.random_state,
                                                                                u, self.unit_dim)
            prior_samples_prime = LatentInvarianceEvaluator.noise_multi_intervention(prior_samples, u, self.unit_dim,
                                                                                     num_interventions=num_interventions, hard=True,
                                                                                     sampling_fun=hybrid_posterior) # m x n x d
            all_prior_samples.append(prior_samples_prime)
        all_prior_samples = torch.vstack(all_prior_samples)# (dxnxm) x d
        responses_prime = self.encode(self.decode(all_prior_samples.to(device), activate=True))
        try: responses_expanded = responses.repeat(num_interventions * num_units, 1)
        except AttributeError:
            # variational case
            _, mus_1, logvars_1 = responses
            responses_expanded = [None,
                                  mus_1.repeat(num_interventions * num_units, 1),
                                  logvars_1.repeat(num_interventions * num_units, 1)]
        errors = self.compute_errors_from_responses(responses_expanded, responses_prime,
                        complete_shape=(num_units, num_interventions, -1, self.latent_size), **kwargs)

        # errors have shape ( u x n x u ) -> the m dimension is reduced in the response computation
        #note: for training we only sum the errors without normalisation --> score not interpretable
        # sum all the errors on non intervened-on dimensions
        errors = errors.mean(dim=1)
        errors = errors*(1.0 - torch.eye(num_units).to(device)) # zeroing out self invariance scores (we don't want any dimension to be invariant to itself)
        errors = errors.mean()
        return errors

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

    def __init__(self, params):
        super(CausalAE, self).__init__(params)

    def compute_errors_from_responses(self, R, R_prime, **kwargs):
        """
        R, R_prime: Tensors of dimension m x D
        Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        unit_dim = kwargs.get('unit_dim',1)
        complete_shape = kwargs.get('complete_shape')
        return LatentInvarianceEvaluator.compute_absolute_errors(R.view(complete_shape),
                                                                 R_prime.view(complete_shape),
                                                                 reduce_dim=2,
                                                                 unit_dim=unit_dim)

class XCSAE(CausalAE, XSAE):

    def __init__(self, params: dict) -> None:
        super(XCSAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool):
        return XSAE.decode(self, noise, activate)

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = XSAE.add_regularisation_terms(self, *args, **kwargs)
        return losses

class XCAE(CausalAE, XAE):

    def __init__(self, params: dict) -> None:
        super(XCAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool):
        return XAE.decode(self, noise, activate)

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = XAE.add_regularisation_terms(self, *args, **kwargs)
        return losses


class CausalVAE(CausalNet, VAE, ABC):
    """ Causally trained version on VAE network - i.e. any net that implements VAEBase"""
    def __init__(self, params:dict):
        super(CausalVAE, self).__init__(params)

    def compute_errors_from_responses(self, R: List[Tensor], R_prime: List[Tensor], **kwargs):
        # errors shape = (dxnxm) x d ---> reduce=False by default nbn
        errors =  LatentInvarianceEvaluator.compute_distributional_errors(R, R_prime,
                                                                          do_KL=kwargs.get('do_KL',True),
                                                                          reduce=kwargs.get('reduce',False),
                                                                          ignore_variance=kwargs.get('ignore_variance',False))
        # complete shape = d x n x m x d --> reduced to d x n x d
        mean_error = errors.view(kwargs.get('complete_shape')).mean(dim=2)
        return mean_error


    def add_regularisation_terms(self, *args, **kwargs):
        losses = VAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = CausalNet.add_regularisation_terms(self, *args, **kwargs)
        return losses

class XCVAE(CausalVAE, Xnet):
    """ XVAE net augmented with causal training """

    def __init__(self, params: dict) -> None:
        super(XCVAE, self).__init__(params)

    def decode(self, noise:Tensor, activate:bool):
        Z = self.causal_block(noise, masks_temperature=self.tau)
        out_init = self.decoder[0](Z).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](out_init)
        if activate: out = self.act(out)
        return out

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalVAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = Xnet.add_regularisation_terms(self, *args, **kwargs)
        return losses
