"""Module containing the set of functions and classes necessary to compute
latent invariance for the different definitions of latent invariance."""

from typing import List

from torch.utils.data import DataLoader
from torch import Tensor
import torch
from models import FCBlock, VAEBase
import numpy as np
from torch.distributions import distribution, Normal, Uniform
from datasets.utils import gen_bar_updater
from models.BASE import GenerativeAE
from models.utils import KL_multiple_univariate_gaussians


class LatentInvarianceEvaluator(object):
    """ Responsible for evaluating drift in the latent space"""

    def __init__(self, model:GenerativeAE, dataloader:DataLoader, mode="X",
                 device="cpu", random_seed=11, verbose=False):
        """mod
        Mode: parameter indicating the structure of the latent space -> the type of interventions to apply
        Available modes = ["X"] """
        self.model = model
        self.model.to(device)
        self.dataloader = dataloader
        self.mode = mode
        self.device = device
        self.source = None; self.num_batches = None
        self.random_state = np.random.RandomState(random_seed)
        self.verbose = verbose
        self.variational = isinstance(self.model, VAEBase)

    def sample_codes_pool(self, num_batches):
        """ Creates memory storage of latent codes"""
        updater = gen_bar_updater()
        source = []
        for i in range(num_batches):
            with torch.no_grad():
                observations, _ = next(iter(self.dataloader))
                #TODO: make sure this encode works (we only want the sample and not the parameters of the distribution)
                codes = self.model.encode(observations.to(self.device))
                if self.variational: codes = codes[0]
                source.append(codes)
                updater(i+1, 1, num_batches)
        self.source = torch.vstack(source); self.N, self.D = self.source.shape
        self.num_batches = num_batches

    """
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
        # General method to perform intervention on latent dimension.
        # Acts as a switch for independent/dependent latents methods
        # Returns tensor of size M x D with 1<= M <=num_samples
        # Note: the order of the latent dimensions in latent_dims and values must be the same.
        #TODO: we could bootstrap here for a more robust estimateof the drift
        if self.independent: return self.intervene_independent(codes, latent_dims, values, num_samples)
        return self.intervene_dependent(codes, latent_dims, values, num_samples, self.device)
    """

    def sample_latent_values(self, codes, dim, n):
        """
        Samples n distinct values from the given latent dimension from the aggregate posterior
        Returns 1 dimensional numpy array with length n
        """
        z = codes[:,dim].view(-1,1).cpu().numpy()
        available_samples = z.shape[0]; n = min(n, available_samples)
        idx = self.random_state.choice(range(available_samples), size=n, replace=False)
        values = z[idx]
        return values

    def causes_intervention(self):
        #TODO
        pass

    @staticmethod
    def noise_intervention(noise, dim:int, hard=True, sampling_fun=None):
        """ Computes hard/soft intervention on the noise variables (i.e. variables that
         are assumed to be independent) """
        if type(noise) == np.ndarray: noise = noise.copy()
        elif type(noise) == torch.Tensor: noise = noise.clone()
        if hard: noise[:,dim] = sampling_fun()
        else: noise[:, dim] = sampling_fun(noise.shape[0])
        return noise

    @staticmethod
    def posterior_distribution(codes, random_seed, dim:int):
        """Obtain posterior distribution over specific dimension"""
        def sampling_fun(size=1):
            idx = random_seed.choice(range(codes.shape[0]), size=size, replace=True) # replace=True --> iid samples
            return codes[idx, dim]
        return sampling_fun

    @staticmethod
    def random_distribution():
        """ Obtain a univariate distribution by transforming Gaussian noise randomly"""
        white_noise = Normal(0, 1)
        #TODO: initialise block differently every time
        random_MLP = FCBlock(10,[10,10,1], torch.nn.ReLU)

        def sample(size_out=1):
            eps = white_noise.sample([size_out, 10])
            with torch.no_grad():
                noise = random_MLP(eps)
            return noise

        return sample

    @staticmethod
    def compute_invariance_score(errors:torch.Tensor, dim:int, average=True):
        """Computes invariance score given a collection (in form of batched tensor) of errors
        on the reconstruction
        Returns:
        Tensor with invariance scores (with respect to maximal distortions)
            - Dx1 torch tensor if average=False
            - 1x1 tensor if average=True"""
        # note that the errors are only positive (L2 distance)
        # ideaòly errors.max() returns the maximum amount of intervention injected
        invariance_total_error = (errors/errors.max()).mean(dim=0) # Dx1
        invariance_total_error[dim] = 1.0 # -> intervention that has been intervened on has no invariance
        if average: invariance_total_error = invariance_total_error.mean() #1x1
        return 1. - invariance_total_error

    @staticmethod
    def compute_absolute_errors(R, R_prime):
        """
        R, R_prime: Tensors of dimension m x D
        Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        m = R.shape[0]
        error = torch.linalg.norm((R-R_prime), ord=2, dim=0)/m # D x 1
        return error

    @staticmethod
    def compute_distributional_errors(R:List[Tensor], R_prime:List[Tensor]):
        """
            R, R_prime: list of tensors of size (D,D, D), with batch size m  -> parametrising Gaussian
            Returns: Dx1 torch tensor of errors (e_i^{j,k})
            FIXME: make distance symmetric
        """
        _, mus_1, logvars_1 = R
        _, mus_2, logvars_2 = R_prime
        all_KLs = KL_multiple_univariate_gaussians(mus_1, mus_2, logvars_1, logvars_2, reduce=False)
        # all_KLs has shape m x D
        error = all_KLs.mean(dim=0) # D x 1
        return error

    def noise_invariance(self, dim:int, **kwargs):
        """ Evaluates invariance of response map to interventions on the noise variable dim
        kwargs accepted keys:
            - num_samples -> number of samples for each intervention
            - num_interventions -> number of interventions to use to compute invariance score
            - device - SE
            - uniform: if desired sampling from uniform for deterministic models
        """

        device = kwargs.get('device','cpu')
        num_samples = kwargs.get('num_samples', 100)
        num_interventions = kwargs.get('num_interventions', 20)


        assert self.source is not None, "Initialise the Evaluator first"

        prior_samples = self.model.sample_noise_from_prior(num_samples, **kwargs).to(device)
        #FIXME: aggregate posterior estimate instead of using only one batch updating it
        posterior_samples = self.source
        responses = self.model.encode(self.model.decode(prior_samples, activate=True))

        errors = torch.zeros(self.model.latent_size, dtype=torch.float).to(device)
        hybrid_posterior = self.posterior_distribution(posterior_samples, self.random_state, dim)

        for intervention_idx in range(num_interventions):

            prior_samples_prime = LatentInvarianceEvaluator.noise_intervention(prior_samples, dim, hard=True, sampling_fun=hybrid_posterior)
            responses_prime = self.model.encode(self.model.decode(prior_samples_prime.to(device), activate=True))

            #FIXME: multi-dimensional units to be considered
            if self.variational: error = self.compute_distributional_errors(responses, responses_prime)
            else: error = self.compute_absolute_errors(responses, responses_prime)
            errors += (error/error.max()) # D x 1

        return errors/num_interventions



    #
    # @staticmethod
    # def compute_drift(z:Tensor, z_hat:Tensor, dims:List[int], average=False, std=False):
    #     """ Defines drift between two tensors.
    #     Both Z and Z_hat are batched (first is batch dimension, second is latents
    #     and it will be averaged over if average is True)
    #
    #     """
    #     #TODO: maybe scaling drift by dimension norm?
    #     drift = torch.abs(torch.hstack([(z - z_hat)[:,d].view(-1,1) for d in dims])) # B x n_dims
    #     if average:
    #         drift = torch.mean(drift, dim=0) # n_dims
    #     if std: return (drift, torch.std(drift, dim=0)) # n_dims
    #     return drift
    #
    #
    # def evaluate_average_intervention_effect(self, latent_dims:List[int], num_interventions,
    #                                                 num_samples, intervention_values=None):
    #     """ Computes the effect of intervention on one or more latent dimensions averaged over several
    #     batches of codes and num_interventions distinct latent values.
    #     Advice:
    #     - keep num_interventions high and num_samples low to ...
    #     """
    #     # Now sample a set of values to use in the intervention
    #     if intervention_values is None:
    #         if self.verbose: print("Sampling values for interventions...")
    #         latent_values = self.sample_latent_values(self.source, latent_dims, num_interventions) # n x l numpy array
    #     else: latent_values = intervention_values
    #     # Obtain intervention for each sample
    #     drifts = []
    #     if self.verbose: print("Sampling from interventional_distribution...")
    #     # shuffling first (it will emulate sampling only if we shuffle)
    #     shuffled_idx = self.random_state.choice(range(self.N), size=self.N, replace=False)
    #     codes = self.source[shuffled_idx]
    #     for v in latent_values:
    #         Bzv = self.intervene_on_latent(codes, latent_dims, v, num_samples)
    #         with torch.no_grad():
    #             Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))
    #         # drift on the interventional distribution given by do(z_l <- v)
    #         drift = self.compute_drift(Bzv, Bzv_hat, latent_dims, average=True) # l x 1 tensor
    #         drifts.append(drift)
    #     drifts = torch.vstack(drifts) # n x l tensor
    #     return drifts, latent_values
    #
    # def evaluate_individual_drifts(self, num_batches, num_interventions, num_samples, reduce=None, resample_codes=False):
    #     """Computes the average intervention effect for multiple interventions on all the latent dimensions.
    #     Returns either a Dx1 vector of scores (reduce = "mean" or "max") , or a matrix of num_interventions x D scores
    #     - @reduce in ["mean", "max", None]
    #     #TODO: probably possible to parallelise here
    #     """
    #     if self.source is None or self.num_batches!=num_batches or resample_codes:
    #         # need to resample the batches of latent codes
    #         if self.verbose: ("Sampling codes from aggregate posterior...")
    #         self.sample_codes_pool(num_batches)
    #
    #     if self.verbose: print("Evaluating individual drifts on latents.")
    #     all_drifts = []; all_interventions = []
    #     for d in range(self.D):
    #         if self.verbose: print(f"Dimension {d}")
    #         drifts, interventions = self.evaluate_average_intervention_effect([d], num_interventions, num_samples)
    #         all_drifts.append(drifts) # n_int x 1
    #         all_interventions.append(interventions)
    #     all_drifts = torch.hstack(all_drifts)
    #     all_interventions = np.hstack(all_interventions) # n_int x d
    #     #finally reducing
    #     if reduce is not None:
    #         if reduce == 'max': all_drifts = torch.max(all_drifts, dim=0)[0]
    #         elif reduce=='mean': all_drifts = torch.mean(all_drifts, dim=0)
    #         else: raise NotImplementedError(f"The reduce modality requested -{reduce}- is not available")
    #     return all_drifts, all_interventions
    #
    # def evaluate_pairwise_drifts(self, num_batches, num_interventions, num_samples, max=False, resample_codes=False):
    #     "Computes pair-wise drifts between all the dimensions and returns a matrix with the results"
    #     if self.source is None or self.num_batches!=num_batches or resample_codes:
    #         # need to resample the batches of latent codes
    #         if self.verbose: ("Sampling codes from aggregate posterior...")
    #         self.sample_codes_pool(num_batches)
    #     if self.verbose: print("Evaluating pairwise drifts...")
    #     matrix = torch.zeros(self.D,self.D)
    #     for i in range(self.D):
    #         for j in range(i):
    #             print(f"Dimensions {i} and {j}")
    #             drifts, _ = self.evaluate_average_intervention_effect([i,j], num_interventions, num_samples)# n_int x 2
    #             m = drifts.max(dim=0)[0] if max else drifts.mean(dim=0)
    #             matrix[i,j] = m[0]; matrix[j,i] = m[1]
    #         # note: here we'll be using a different set of interventions from those used above
    #         # For small number of interventions the estimate is more precise if we compare on the same
    #         # set of interventions. However, the increase in computational cost would be significant
    #         drifts, _ = self.evaluate_average_intervention_effect([i], num_interventions, num_samples)# n_int x 1
    #         matrix[i,i] = drifts.max() if max else drifts.mean()
    #     for i in range(self.D):
    #         # now the i,j entry of the matrix is measuring the additional drift encountered
    #         # by intervening also on j
    #         matrix[i,:] = (matrix[i,:] - matrix[i,i])/(matrix[i,i] +10e-7)
    #     return matrix
    #
    # def evaluate_effect_on_means(self, latent_dims:List[int], num_interventions, num_samples, intervention_values=None):
    #     """ Computes the effect of intervention on one or more latent dimensions on the mean of the latent dimension.
    #     Advice:
    #     - keep num_interventions high and num_samples low to ...
    #     """
    #
    #     # Now sample a set of values to use in the intervention
    #     if intervention_values is None:
    #         if self.verbose: print("Sampling values for interventions...")
    #         latent_values = self.sample_latent_values(self.source, latent_dims, num_interventions) # n x l numpy array
    #     else: latent_values = intervention_values
    #     # Obtain intervention for each sample
    #     means = []
    #     if self.verbose: print("Sampling from interventional_distribution...")
    #     # shuffling first (it will emulate sampling only if we shuffle)
    #     shuffled_idx = self.random_state.choice(range(self.N), size=self.N, replace=False)
    #     codes = self.source[shuffled_idx]
    #     for v in latent_values:
    #         Bzv= self.intervene_on_latent(codes, latent_dims, v, num_samples) # n x l
    #         with torch.no_grad():
    #             Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))
    #             means.append(Bzv_hat.mean(dim=0)[latent_dims]) # l x 1
    #     means = torch.vstack(means) # Ni x l tensor
    #     return means, latent_values
    #
    # def compute_normalisation_constant(self, dim, num_interventions, num_samples, intervention_values=None):
    #     """ Computes normalisation constant of the dimension 'dim' scoring as the maximum
    #     distortion of the mean caused by an intervention on it.
    #     Note: to be used for IRS scoring"""
    #
    #     # Now sample a set of values to use in the intervention
    #     if intervention_values is None:
    #         if self.verbose: print("Sampling values for interventions...")
    #         latent_values = self.sample_latent_values(self.source, [dim], num_interventions) # n x 1 numpy array
    #     else: latent_values = intervention_values
    #
    #     if self.verbose: print(f"Sampling from interventional_distributions on {dim}...")
    #     # shuffling first (it will emulate sampling only if we shuffle)
    #     shuffled_idx = self.random_state.choice(range(self.N), size=self.N, replace=False)
    #     codes = self.source[shuffled_idx]
    #     means_distortion = []
    #     originals_mean = None
    #     for v in latent_values:
    #         Bzv = self.intervene_on_latent(codes, [dim], v, num_samples); n = Bzv.shape[0] # n x D
    #         with torch.no_grad():
    #             if originals_mean is None:
    #                 originals_hat = self.model.encode_mu(self.model.decode(codes[:n].to(self.device), activate=True))
    #                 originals_mean = originals_hat[:,dim].mean()
    #             Bzv_hat = self.model.encode_mu(self.model.decode(Bzv.to(self.device), activate=True))
    #         means_distortion.append(Bzv_hat[:,dim].mean() - originals_mean)
    #     means_distortion = torch.max(torch.stack(means_distortion)).item()
    #     return means_distortion
    #
    # def evaluate_pairwise_effects(self, num_batches, num_interventions, num_samples, resample_codes=False, max=False):
    #     """Computes pair-wise effects on means for each pair of ltent dimensions"""
    #     if self.source is None or self.num_batches!=num_batches or resample_codes:
    #         # need to resample the batches of latent codes
    #         if self.verbose: print("Sampling codes from aggregate posterior...")
    #         self.sample_codes_pool(num_batches)
    #     if self.verbose: print("Evaluating pairwise means effects...")
    #     matrix = torch.zeros(self.D,self.D)
    #     for i in range(self.D):
    #         for j in range(i):
    #             print(f"Dimensions {i} and {j}")
    #             # ---- pair (i,j) effects
    #             muij, interventions = self.evaluate_effect_on_means([i,j], num_interventions, num_samples)# n_int x 2
    #             # ---- i effect
    #             zi = interventions[:,0].reshape(-1,1)
    #             mui, _zi = self.evaluate_effect_on_means([i], num_interventions, num_samples, intervention_values = zi)# n_int x 1
    #             assert np.array_equal(zi,_zi)
    #             delta = (muij[:,0] - mui[:,0]).abs()
    #             matrix[i,j] = delta.max() if max else delta.mean()
    #             # ---- j effect
    #             zj = interventions[:,1].reshape(-1,1)
    #             muj, _zj = self.evaluate_effect_on_means([j], num_interventions, num_samples, intervention_values = zj)# n_int x 1
    #             assert np.array_equal(zj,_zj)
    #             delta = (muij[:,1] - muj[:,0]).abs()
    #             matrix[j,i] = delta.max() if max else delta.mean()
    #     for i in range(self.D):
    #         print(f"Computing normalisation constant for {i}...")
    #         Z = self.compute_normalisation_constant(i, num_interventions, num_samples)
    #         matrix[i,:] = matrix[i,:]/Z
    #     return matrix
    #
    #
    #
    #






