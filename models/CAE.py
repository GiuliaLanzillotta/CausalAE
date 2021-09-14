"""Set of CAUSAL autoencoder models, i.e. models trained to be equivariant/invariant to interventions in the latent space."""

from abc import abstractmethod, ABC
from typing import List

import copy
import numpy as np
import torch
from torch import Tensor, nn

from metrics import LatentConsistencyEvaluator
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

    @abstractmethod
    def compute_errors_from_causal_vars(self, X1, X2, **kwargs):
        """ X1 and X2 have the same shape, namely (l x d) """
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
            prior_samples = self.sample_noise_from_prior(num_samples, prior_mode=prior_mode, device=device)
            #Note that we're using only the available batch to approximaate the aggregate posterior estimate
            posterior_samples = self.sample_noise_from_posterior(inputs, device=device)
            # note: the output of the 'encode' method could be whatever (e.g. a list, a Tensor)
            responses = self.encode(self.decode(prior_samples, activate=True)) # n x d


        all_prior_samples = []
        num_units = self.latent_size//self.unit_dim
        for u in range(num_units):
            hybrid_posterior = LatentConsistencyEvaluator.posterior_distribution(posterior_samples.detach(),
                                                                                 self.random_state,
                                                                                 u, self.unit_dim)
            prior_samples_prime = LatentConsistencyEvaluator.noise_multi_intervention(prior_samples, u, self.unit_dim,
                                                                                      num_interventions=num_interventions,
                                                                                      hard=True,
                                                                                      sampling_fun=hybrid_posterior) # m x n x d
            all_prior_samples.append(prior_samples_prime)
        all_prior_samples = torch.vstack(all_prior_samples)# (dxnxm) x d
        responses_prime = self.encode(self.decode(all_prior_samples.to(device), activate=True))
        try: responses_expanded = responses.detach().repeat(num_interventions * num_units, 1)
        except AttributeError:
            # variational case
            _, mus_1, logvars_1 = responses
            responses_expanded = [None,
                                  mus_1.detach().repeat(num_interventions * num_units, 1),
                                  logvars_1.detach().repeat(num_interventions * num_units, 1)]
        errors = self.compute_errors_from_responses(responses_expanded, responses_prime,
                        complete_shape=(num_units, num_interventions, -1, self.latent_size), **kwargs)

        # errors have shape ( u x n x u ) -> the m dimension is reduced in the response computation
        #note: for training we only sum the errors without normalisation --> score not interpretable
        # sum all the errors on non intervened-on dimensions
        errors = errors.mean(dim=1)
        errors = errors*(1.0 - torch.eye(num_units).to(device)) # zeroing out self invariance scores (we don't want any dimension to be invariant to itself)
        errors = errors.mean()
        return errors

    def compute_equivariance_loss(self, inputs, **kwargs):
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
        latent_size_prime = kwargs.get('latent_size_prime',self.latent_size)


        with torch.no_grad():
            prior_samples = self.sample_noise_from_prior(num_samples, prior_mode=prior_mode, device=device)
            #Note that we're using only the available batch to approximaate the aggregate posterior estimate
            posterior_samples = self.sample_noise_from_posterior(inputs, device=device)

        # note: the output of the 'encode' method could be whatever (e.g. a list, a Tensor)
        #TODO: generalise to the whole distribution
        responses = self.encode_mu(self.decode(prior_samples, activate=True)) # n x d


        all_prior_samples = []
        all_responses_prime = []

        for d in range(self.latent_size):
            hybrid_posterior = LatentConsistencyEvaluator.posterior_distribution(posterior_samples, self.random_state, d, unit_dim=1)
            all_samples = torch.vstack([prior_samples, responses]) #2m x d
            all_samples_prime = LatentConsistencyEvaluator.noise_multi_intervention(all_samples, d, unit_dim=1,  # here if model is variational we need to sample multiple times from the intervened response
                                                                                    num_interventions=num_interventions,
                                                                                    hard=True, sampling_fun=hybrid_posterior) # m x n x d
            all_samples_prime = all_samples_prime.view(num_interventions, num_samples*2, self.latent_size)
            prior_samples_prime = all_samples_prime[:,:num_samples,:]; responses_prime = all_samples_prime[:,num_samples:,:]
            all_prior_samples.append(prior_samples_prime.contiguous().view(num_samples*num_interventions, self.latent_size))#(nxm) x d
            all_responses_prime.append(responses_prime.contiguous().view(num_samples*num_interventions, self.latent_size)) #(nxm) x d

        all_prior_samples = torch.vstack(all_prior_samples)# (dxnxm) x d
        all_responses_prime = torch.vstack(all_responses_prime)# (dxnxm) x d
        all_responses_prime2 = self.encode_mu(self.decode(all_prior_samples.to(device), activate=True))
        # now get causal variables from both
        X_prime1 = self.get_causal_variables(all_responses_prime, **kwargs)
        X_prime2 = self.get_causal_variables(all_responses_prime2, **kwargs)


        errors = self.compute_errors_from_causal_vars(X_prime1, X_prime2,
                                                      complete_shape=(self.latent_size, num_interventions,
                                                                      -1, latent_size_prime), **kwargs)
        # errors will have shape latent_size x n x latent_size (latent_size_prim/xunit_dim)
        errors = errors.mean()
        return errors

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        equivariance = kwargs.get('equivariance',False)
        losses = kwargs.get('losses')
        lamda = kwargs.get('invariance_lamda')
        X = kwargs["X"]
        if equivariance:
            L_reg = self.compute_equivariance_loss(inputs = X, **kwargs)
            losses['Equivariance_loss'] = L_reg
        else:
            L_reg = self.compute_invariance_loss(inputs = X, **kwargs)
            losses['Invariance_loss'] = L_reg
        losses['loss'] += lamda*L_reg
        return losses


class CausalAE(CausalNet, ABC):
    """Causally trained version of the HybridAE: simply adds a regularisation term
    to the reconstruction objective."""

    def __init__(self, params):
        super(CausalAE, self).__init__(params)

    def compute_errors_from_causal_vars(self, X1, X2, **kwargs):
        """ X1 and X2 have the same shape, namely (l x d) """
        xunit_dim = kwargs.get('xunit_dim',1)
        complete_shape = kwargs.get('complete_shape')
        return LatentConsistencyEvaluator.compute_absolute_errors(X1.view(complete_shape),
                                                                  X2.view(complete_shape),
                                                                  reduce_dim=2,
                                                                  unit_dim=xunit_dim)

    def compute_errors_from_responses(self, R, R_prime, **kwargs):
        """
        R, R_prime: Tensors of dimension m x D
        Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        unit_dim = kwargs.get('unit_dim',1)
        complete_shape = kwargs.get('complete_shape')
        return LatentConsistencyEvaluator.compute_absolute_errors(R.view(complete_shape),
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

    def compute_errors_from_causal_vars(self, X1, X2, **kwargs):
        """ X1 and X2 have the same shape, namely (l x d)
        Note: this only works with the current setting:
        i.e. only requiring equivariance for the mean of the posteriors"""
        unit_dim = kwargs.get('unit_dim',1)
        complete_shape = kwargs.get('complete_shape')
        return LatentConsistencyEvaluator.compute_absolute_errors(X1.view(complete_shape),
                                                                  X2.view(complete_shape),
                                                                  reduce_dim=2,
                                                                  unit_dim=unit_dim)

    def compute_errors_from_responses(self, R: List[Tensor], R_prime: List[Tensor], **kwargs):
        # errors shape = (dxnxm) x d ---> reduce=False by default nbn
        errors =  LatentConsistencyEvaluator.compute_distributional_errors(R, R_prime,
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

    def decode_from_X(self, x, *args, **kwargs):
        activate = kwargs.get('activate', False)
        out_init = self.decoder[0](x).view((-1, )+self.decoder_initial_shape) # reshaping into image format
        out = self.decoder[1](out_init)
        if activate: out = self.act(out)
        return out

    def add_regularisation_terms(self, *args, **kwargs):
        losses = CausalVAE.add_regularisation_terms(self, *args, **kwargs)
        kwargs['losses'] = losses
        losses = Xnet.add_regularisation_terms(self, *args, **kwargs)
        return losses
