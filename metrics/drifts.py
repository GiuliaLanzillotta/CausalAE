"""Code to provide estimates of latent drift
--- Part of the causal modelling framework --- """
from typing import List

from torch.utils.data import DataLoader
from torch import Tensor
import torch
import numpy as np

from datasets.utils import gen_bar_updater
from models.BASE import GenerativeAE




class DriftEvaluator(object):
    """ Responsible for evaluating drift in the latent space"""

    def __init__(self, model:GenerativeAE, dataloader:DataLoader, independent:bool,
                 drift_norm = 1, device="cpu", random_seed=11):
        """ Options to consider for drift_norm parameter include: 0,1,2"""
        self.model = model
        self.dataloader = dataloader
        self.independent = independent
        self.drift_norm = drift_norm
        self.device = device
        self.source = None; self.num_batches = None
        self.random_state = np.random.RandomState(random_seed)

    def sample_codes_pool(self, num_batches):
        """ Creates memory storage of latent codes"""
        updater = gen_bar_updater()
        source = []
        for i in range(num_batches):
            with torch.no_grad():
                observations, _ = next(iter(self.dataloader))
                codes = self.model.encode_mu(observations.to(self.device))
                source.append(codes)
                updater(i+1, 1, num_batches)
        self.source = torch.vstack(source); self.N, self.D = self.source.shape
        self.num_batches = num_batches

    @staticmethod
    def intervene_dependent(codes:Tensor, latent_dims:List[int], values, num_samples, device):
        final_condition = torch.ones_like(codes[:,0]) # all rows
        for i,d in enumerate(latent_dims):
            condition = codes[:,d] == values[i].item()
            final_condition = final_condition*condition
        intervened_codes = codes.detach().clone()[final_condition.bool().to(device)][:num_samples]
        return intervened_codes

    @staticmethod
    def intervene_independent(codes:Tensor, latent_dims:List[int], values, num_samples) -> Tensor:
        intervened_codes = codes.detach().clone()[:num_samples]

        for i,d in enumerate(latent_dims):
            intervened_codes[:,d] = values[i].item()
        return intervened_codes

    def intervene_on_latent(self, codes, latent_dims:List[int], values, num_samples):
        """ General method to perform intervention on latent dimension.
        Acts as a switch for independent/dependent latents methods
        Returns tensor of size M x D with 1<= M <=num_samples
        Note: the order of the latent dimensions in latent_dims and values must be the same.
        """
        #TODO: we could bootstrap here for a more robust estimateof the drift
        # shuffling first (it will emulate sampling only if we shuffle)
        shuffled_idx = self.random_state.choice(range(self.N), size=self.N, replace=False)
        codes = codes[shuffled_idx]
        if self.independent: return self.intervene_independent(codes, latent_dims, values, num_samples), codes
        return self.intervene_dependent(codes, latent_dims, values, num_samples, self.device), codes

    def sample_latent_values(self, codes, dims, n):
        """
        Samples n distinct values from the given latent dimension
        l = len(dims)
        Returns n x l numpy 2-dimensional array"""
        z = np.unique(torch.hstack([codes[:,d].view(-1,1) for d in dims]).cpu().numpy(), axis=0)
        available_samples = z.shape[0]; n = min(n, available_samples)
        idx = self.random_state.choice(range(available_samples), size=n, replace=False)
        values = z[idx]
        return values.reshape(-1, len(dims))


    @staticmethod
    def compute_drift(z:Tensor, z_hat:Tensor, dims:List[int], average=False, std=False):
        """ Defines drift between two tensors.
        Both Z and Z_hat are batched (first is batch dimension, second is latents
        and it will be averaged over if average is True)

        """
        #TODO: maybe scaling drift by dimension norm?
        drift = torch.abs(torch.hstack([(z - z_hat)[:,d].view(-1,1) for d in dims])) # B x n_dims
        if average:
            drift = torch.mean(drift, dim=0) # n_dims
        if std: return (drift, torch.std(drift, dim=0)) # n_dims
        return drift


    def evaluate_average_intervention_effect(self, latent_dims:List[int], num_batches, num_interventions,
                                             num_samples, resample_codes=False, norm=True, intervention_values=None):
        """ Computes the effect of intervention on one or more latent dimensions averaged over several
        batches of codes and num_interventions distinct latent values.
        Advice:
        - keep num_interventions high and num_samples low to ...
        """
        if self.source is None or self.num_batches!=num_batches or resample_codes:
            # need to resample the batches of latent codes
            print("Sampling codes from aggregate posterior...")
            self.sample_codes_pool(num_batches)
        # Now sample a set of values to use in the intervention
        if intervention_values is None:
            print("Sampling values for interventions...")
            latent_values = self.sample_latent_values(self.source, latent_dims, num_interventions) # n x l numpy array
        else: latent_values = intervention_values
        # Obtain intervention for each sample
        drifts = []
        print("Sampling from interventional_distribution...")
        for v in latent_values:
            Bzv, _ = self.intervene_on_latent(self.source, latent_dims, v, num_samples)
            with torch.no_grad():
                Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))
            # drift on the interventional distribution given by do(z_l <- v)
            drift = self.compute_drift(Bzv, Bzv_hat, latent_dims, average=True) # l x 1 tensor
            drifts.append(drift)
        drifts = torch.vstack(drifts) # n x l tensor
        if norm: return torch.linalg.norm(drifts, ord=self.drift_norm, dim=0)
        return drifts, latent_values


    def evaluate_effect_on_means(self, latent_dims:List[int], num_batches, num_interventions, num_samples,
                                 resample_codes=False, intervention_values=None):
        """ Computes the effect of intervention on one or more latent dimensions on the mean of the latent dimension.
        Advice:
        - keep num_interventions high and num_samples low to ...
        """
        if self.source is None or self.num_batches!=num_batches or resample_codes:
            # need to resample the batches of latent codes
            print("Sampling codes from aggregate posterior...")
            self.sample_codes_pool(num_batches)
        # Now sample a set of values to use in the intervention
        if intervention_values is None:
            print("Sampling values for interventions...")
            latent_values = self.sample_latent_values(self.source, latent_dims, num_interventions) # n x l numpy array
        else: latent_values = intervention_values
        # Obtain intervention for each sample
        means = []
        print("Sampling from interventional_distribution...")
        for v in latent_values:
            Bzv, _ = self.intervene_on_latent(self.source, latent_dims, v, num_samples) # n x l
            with torch.no_grad():
                Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))
                means.append(Bzv_hat.mean(dim=0)[latent_dims]) # l x 1
        means = torch.vstack(means) # Ni x l tensor
        return means, latent_values

    def compute_normalisation_constant(self, dim, num_interventions, num_samples, drifts=False):
        """ Computes normalisation constant of the dimension 'dim' scoring as the maximum
        distortion caused by an intervention on any dimension on it.
        Note: it does not resample codes:assumes that pool of codes has already been sampled"""

        # collecting distortion from every dimension
        all_distortions = []
        for d in range(self.D):
            print(f"Sampling values for interventions on {d}...")
            latent_values = self.sample_latent_values(self.source, [d], num_interventions) # n x l numpy array

            print(f"Sampling from interventional_distributions from {d}...")
            distortions = 0.0 # we collect the sum of abolute distortions over the interventional distributions
            for v in latent_values:
                Bzv, originals = self.intervene_on_latent(self.source, [d], v, num_samples) # n x D
                n = Bzv.shape[0]; originals = originals[:n]
                with torch.no_grad():
                    originals_hat = self.model.encode_mu(self.model.decode(originals.to(self.device), activate=True))
                    Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))

                if not drifts: distortions += (originals_hat[:,dim].mean() - Bzv_hat[:,dim].mean()).item()
                else: distortions += self.compute_drift(originals, Bzv_hat, [dim], average=True).item() # 1x1
            all_distortions.append(distortions/num_interventions) # taking the average
        all_distortions = np.array(all_distortions)
        return np.max(all_distortions)












