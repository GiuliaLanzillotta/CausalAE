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
from models.utils import KL_multiple_univariate_gaussians, distribution_parameter_distance


class LatentConsistencyEvaluator(object):
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

    def sample_codes_pool(self, num_batches, device):
        """ Creates memory storage of latent codes by sampling from the posterior distribution.
        Approximation of the aggregate posterior distribution. """
        updater = gen_bar_updater()
        source = []
        for i in range(num_batches):
            with torch.no_grad():
                observations, _ = next(iter(self.dataloader))
                codes = self.model.sample_noise_from_posterior(observations.to(device), device=device)
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
    def noise_intervention(noise, unit:int, unit_dim:int, hard=True, sampling_fun=None):
        """ Computes hard/soft intervention on the noise variables (i.e. variables that
         are assumed to be independent) """
        m,_ = noise.shape
        if type(noise) == np.ndarray: noise = torch.from_numpy(noise.copy())
        elif type(noise) == torch.Tensor: noise = noise.detach().clone()
        if hard: intv = sampling_fun() # only one value
        else: intv = sampling_fun(m)
        noise[:,unit*unit_dim:(unit+1)*unit_dim] = intv
        return noise

    @staticmethod
    def noise_multi_intervention(noise, unit:int, unit_dim:int, num_interventions:int, hard=True, sampling_fun=None):
        """ Computes hard/soft intervention on the noise variables (i.e. variables that
         are assumed to be independent) """
        device = noise.device
        m,d = noise.shape
        if type(noise) == np.ndarray: noise = torch.from_numpy(noise.copy())
        elif type(noise) == torch.Tensor: noise = noise.detach().clone()
        # noise has shape m x d
        # we want it to have shape n x m x d
        noise = torch.tile(noise,(num_interventions, 1,1))
        if hard: intv = sampling_fun(num_interventions).repeat(m,1) # (mxn) x D_u   - same intervention value for all m samples
        else: intv = sampling_fun(m*num_interventions) # (mxn) x D_u
        noise[:,:,unit*unit_dim:(unit+1)*unit_dim] = intv.view(m, num_interventions, unit_dim).permute(1,0,2)
        return noise.view(-1, d) # (mxn) x d

    @staticmethod
    def posterior_distribution(codes, random_seed, unit_num:int, unit_dim:int):
        """Obtain posterior distribution over specific dimension"""
        def sampling_fun(size=1):
            idx = random_seed.choice(range(codes.shape[0]), size=size, replace=True) # replace=True --> iid samples
            return codes[idx, unit_num*unit_dim:(unit_num+1)*unit_dim] # m x u_d
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
    def compute_absolute_errors(R, R_prime, reduce_dim:int=0, unit_dim:int=1):
        """
        R, R_prime: Tensors of dimension u x n x m x D
        Returns: Dx1 torch tensor of errors (e_i^{j,k})"""
        delta = R-R_prime
        # reshaping to extract the single units
        delta = delta.view(delta.shape[:-1] + (-1,unit_dim)) #expanding last dimension to separate units
        error = torch.linalg.norm(delta, ord=2, dim=-1).mean(dim=reduce_dim) # num_units x 1
        return error

    @staticmethod
    def compute_distributional_errors(R:List[Tensor], R_prime:List[Tensor], do_KL=False, reduce=False, ignore_variance=False):
        """
            R, R_prime: list of tensors of size (D,D, D), with batch size m  -> parametrising Gaussian
            Returns: Dx1 torch tensor of errors (e_i^{j,k})
            FIXME: make distance symmetric
        """
        _, mus_1, logvars_1 = R
        _, mus_2, logvars_2 = R_prime
        if do_KL: error = KL_multiple_univariate_gaussians(mus_1, mus_2, logvars_1, logvars_2, reduce=reduce)
        else: error = distribution_parameter_distance(mus_1, mus_2, logvars_1, logvars_2,
                                                      reduce=reduce, ignore_variance=ignore_variance)
        return error

    def compute_std_dev(self, prior_samples, responses, num_units:int, unit_dim:int):
        """Takes samples from the prior and second aggregate posterior
        distributions and returns the standard deviation measured per latent unit."""

        if self.variational: # encode returns a list of parameters
            responses=responses[1]
        # shape = 2 x num_units x 1
        std_dev = torch.vstack([torch.std(prior_samples.view(-1, num_units, unit_dim), dim=[0,2]),
                                torch.std(responses.view(-1, num_units, unit_dim), dim=[0,2])]).detach()

        return std_dev


    def self_consistency(self, unit_dim:int, num_units:int, num_samples:int, **kwargs):
        """Equivariance of response map under null intervention
            - num_samples -> number of samples for each intervention
            - device - SE
            - uniform: if desired sampling from uniform for deterministic models
            - mode: prior sampling mode (for non variational models): hybrid/uniform/posterior"""

        assert self.source is not None, "Initialise the Evaluator first"
        prior_samples = self.model.sample_noise_from_prior(num_samples, **kwargs)
        responses = self.model.encode_mu(self.model.decode(prior_samples, activate=True)).detach()
        #observed standard deviation on each latent unit
        std_dev = self.compute_std_dev(prior_samples, responses, num_units, unit_dim)
        errors = self.compute_absolute_errors(prior_samples, responses, reduce_dim=0, unit_dim=unit_dim).detach()
        return errors, std_dev # average error over interventions

    def noise_equivariance(self, unit:int, xunit_dim:int,  num_samples:int, **kwargs):
        """ Evaluates equivariance of response map to interventions on the noise variable dim
        kwargs accepted keys:
            - num_samples -> number of samples for each intervention
            - num_interventions -> number of interventions to use to compute invariance score
            - device - SE
            - prior_mode: prior sampling mode (for non variational models): hybrid/uniform/posterior
        """

        assert self.source is not None, "Initialise the Evaluator first"
        device = kwargs.get('device','cpu')
        num_interventions = kwargs.get('num_interventions', 20)

        prior_samples = self.model.sample_noise_from_prior(num_samples, **kwargs)
        responses = self.model.encode_mu(self.model.decode(prior_samples, activate=True)).detach()
        # same intervention must be applied to prior distribution and response posterior
        all_samples = torch.vstack([prior_samples, responses]) # shape = (m x 2, d)

        posterior_samples = self.source
        hybrid_posterior = self.posterior_distribution(posterior_samples, self.random_state, unit, 1)

        errors = torch.zeros(self.model.latent_size, dtype=torch.float, device=device) #TODO: check the sizes here when increasing number of causal units

        for intervention_idx in range(num_interventions):
            all_samples_prime = self.noise_intervention(all_samples, unit, 1, hard=True, sampling_fun=hybrid_posterior).to(device)
            prior_samples_prime = all_samples_prime[:num_samples]; responses_prime = all_samples_prime[num_samples:]
            responses_prime2 = self.model.encode_mu(self.model.decode(prior_samples_prime, activate=True)).detach() # m x d
            # now get causal variables from both
            X_prime1 = self.model.get_causal_variables(responses_prime, **kwargs)# Dx x 1
            X_prime2 = self.model.get_causal_variables(responses_prime2, **kwargs)# Dx x 1
            # we're ignoring the variance to make the results from VAE comparable with results from Hybrid models
            error = self.compute_absolute_errors(X_prime1, X_prime2, reduce_dim=0, unit_dim=xunit_dim) # D x 1

            intervention_magnitude = torch.linalg.norm((prior_samples_prime-prior_samples), ord=2, dim=1).mean() # mean over samples of the delta
            errors += (error/(intervention_magnitude + 10e-5)) # D x 1

        #TODO: take the meadian instead of mean when averaging over interventions
        # OR store the worst intervention and return it
        return errors/num_interventions # average error over interventions


    def noise_invariance(self, unit:int, unit_dim:int, num_units:int, num_samples:int, **kwargs):
        """ Evaluates invariance of response map to interventions on the noise variable dim
        kwargs accepted keys:
            - num_samples -> number of samples for each intervention
            - num_interventions -> number of interventions to use to compute invariance score
            - device - SE
            - uniform: if desired sampling from uniform for deterministic models
            - mode: prior sampling mode (for non variational models): hybrid/uniform/posterior
        """

        device = kwargs.get('device','cpu')
        num_interventions = kwargs.get('num_interventions', 20)

        assert self.source is not None, "Initialise the Evaluator first"

        prior_samples = self.model.sample_noise_from_prior(num_samples, **kwargs).detach()
        posterior_samples = self.source
        responses = self.model.encode(self.model.decode(prior_samples, activate=True))

        #observed standard deviation on each latent unit
        std_dev = self.compute_std_dev(prior_samples, responses, num_units, unit_dim)

        errors = torch.zeros(num_units, dtype=torch.float, device=device)
        hybrid_posterior = self.posterior_distribution(posterior_samples, self.random_state, unit, unit_dim)

        for intervention_idx in range(num_interventions):
            prior_samples_prime = self.noise_intervention(prior_samples, unit, unit_dim, hard=True, sampling_fun=hybrid_posterior).detach()
            responses_prime = self.model.encode(self.model.decode(prior_samples_prime.to(device), activate=True)) # m x d
            # we're ignoring the variance to make the results from VAE comparable with results from Hybrid models
            if self.variational: error = self.compute_distributional_errors(responses, responses_prime,
                                                                            reduce=False, do_KL=False,
                                                                            ignore_variance=True).mean(dim=0) # m x D --> D x 1
            else: error = self.compute_absolute_errors(responses, responses_prime, reduce_dim=0, unit_dim=unit_dim) # D x 1

            # normalisation across spaces -> perfect invariance (i.e. 1.0) only obtained if the different spaces have same scale
            intervention_magnitude = torch.linalg.norm((prior_samples_prime-prior_samples), ord=2, dim=1).mean() # mean over samples of the delta
            errors += (error/(intervention_magnitude + 10e-5)) # D x 1

        #TODO: take the meadian instead of mean when averaging over interventions
        # OR store the worst intervention and return it
        return errors/num_interventions, std_dev # average error over interventions



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






