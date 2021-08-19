"""Set of CAUSAL autoencoder models, i.e. models trained to be equivariant/invariant to interventions in the latent space."""

from abc import abstractmethod, ABC

import torch
from torch import Tensor

import numpy as np

from . import HybridAE
from metrics import LatentInvarianceEvaluator


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
        return  output

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
        num_samples = kwargs.get('num_samples', 512) # advice- use num sam
        use_MSE = kwargs.get('use_MSE',True)

        MSE,BCE = self.pixel_losses(X,X_hat)
        L_rec = MSE if use_MSE else BCE

        latent_samples = self.hybrid_layer.sample_from_prior(num_samples)
        responses = self.encode(self.decode(latent_samples.to(device), activate=True))

        #TODO: parallelise to make faster

        invariance_sum = 0.

        for d in range(self.latent_size):
            hybrid_posterior = LatentInvarianceEvaluator.posterior_distribution(latent_samples, self.random_state, d)
            latent_samples_prime = LatentInvarianceEvaluator.noise_intervention(latent_samples, d, hard=True, sampling_fun=hybrid_posterior)
            responses_prime = self.encode(self.decode(latent_samples_prime.to(device), activate=True))
            errors = torch.linalg.norm((responses-responses_prime), ord=2, dim=0)/num_samples # D x 1
            errors = (errors/errors.max())
            # sum all the errors on non intervened-on dimensions
            invariance_sum += torch.sum(errors[:d]) + torch.sum(errors[d+1:])

        L_reg = invariance_sum
        loss = L_rec + lamda*invariance_sum
        return{'loss': loss, 'Reconstruction_loss':L_rec, 'Regularization_loss':L_reg}

