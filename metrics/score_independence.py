""" Module responsible for evaluating a model with (potentially imultple) independence metrics"""
import torch

from . import DCI, BetaVAE, IRS, MIG, ModExp, SAP
import numpy as np
from torch.utils.data import DataLoader
from models.BASE import GenerativeAE
from models.layers import HybridLayer
from models import utils
from torchvision.transforms import Normalize
import dcor

def compute_dCov_multiDim(data, unbiased=False):
    """ Computes the distance covariance over multiple dimensions by aggregating
    the result on pair-wise correlation measures.
    data: numpy array of size NxD
    covariance vs correlation:
    - covariance is faster and safer (no divisions)
    - correlation is normalised
    """
    D = data.shape[1]
    total_corr = 0.0
    count = 0
    for i in range(D):
        for j in range(i+1):
            total_corr+= dcor.distance_covariance(data[:,i], data[:,j]) #note that here we're using the biased estimator
            if unbiased: total_corr+=np.sqrt(dcor.u_distance_covariance_sqr(data[:,i],data[:,j]))
            count+=1
    total_corr/=count #averaging correlations among features
    return total_corr


class LatentOrthogonalityEvaluator(object):
    """ Takes a (trained) generative model and  evaluates the independence between its latent dimensions
    by measuring the degree of invariance to hybridisation. """
    def __init__(self, model:GenerativeAE, test_dataloader:DataLoader, latent_size, unit_dim):
        self.model = model
        self.dataloader = test_dataloader
        self.test_input, _ = next(iter(test_dataloader))
        # Fix the random seed for reproducibility.
        self.random_state = np.random.RandomState(0)
        self.hybridiser = HybridLayer(latent_size, unit_dim, self.test_input.shape[0])

    def score_latent(self, device:str='cpu', **kwargs):
        num_batches = 10
        originals = []
        hybrids = []

        print("Preparing the model for scoring orthogonality ...")
        for i in range(num_batches):
            observations, _ = next(iter(self.dataloader))
            with torch.no_grad():
                codes = self.model.encode_mu(observations.to(device)); originals.append(codes)
                hybridised_codes = self.hybridiser.controlled_sampling(codes, use_prior=False,
                                            hybridisation_level=self.hybridiser.max_hybridisation_level)
            hybrids.append(hybridised_codes)
        originals = torch.vstack(originals); hybrids = torch.vstack(hybrids)

        # normalisation to not take into account the "natural spread" of the latent dstribution
        # when scoring orthogonality
        normls = Normalize(0,1)
        ON = normls(originals.permute(1,0).unsqueeze(2)).squeeze(2).T
        HN = normls(hybrids.permute(1,0).unsqueeze(2)).squeeze(2).T

        print("Scoring orthogonality")
        scores = {}

        with torch.no_grad():
            print("RBF scoring"); scores["RBF"] = float(utils.MMD(*utils.RBF_kernel(ON, HN, device)).cpu().numpy())
            print("IMQ scoring"); scores["IMQ"] = float(utils.IMQ_kernel(ON, HN, device).cpu().numpy())
            print("CAT scoring"); scores["CAT"] = float(utils.MMD(*utils.Categorical_kernel(ON, HN, device,
                                                kwargs.get("strict",True), kwargs.get("hierarchy",True))).cpu().numpy())
            print("dCov scoring"); scores["dCOV"] = compute_dCov_multiDim(ON.cpu().numpy())

        return scores