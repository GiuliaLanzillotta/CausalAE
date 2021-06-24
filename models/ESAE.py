#Code for E(quivariant)SAE experiment
#Experiment number one on SAE independence
import numpy as np
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE, SAE, utils
from torch.nn import functional as F
from .utils import act_switch

class ESAE(SAE):
    """Equivariant version of the SAE"""
    def __init__(self, params:dict, dim_in) -> None:
        super(ESAE, self).__init__(params, dim_in)
        self.max_hybridisation_level = self.latent_size//self.unit_dim
        # M = number of hybrid samples to be drawn during forward pass
        # note that the minimum number of hybrid samoles coincides with the
        # number of hybridisation levels
        #TODO: allow M
        #self.M = max(params["num_hybrid_samples"], self.latent_units)

    def sample_noise_controlled(self, latent_vecs: Tensor, level:int):
        noise = self.hybrid_layer.controlled_sampling(latent_vecs, level, use_prior=False)
        return noise

    def sample_noise_from_posterior(self, inputs: Tensor):
        """Equivalent to performing full hybridisation"""
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes)
        noise = self.hybrid_layer(codes).to(codes.device)
        return noise

    def generate(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        inputs = x.view((-1, )+self.dim_in) # just making sure
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes)
        output = self.decode(codes, activate)
        return  output

    def forward(self, inputs: Tensor, activate:bool=False, update_prior:bool=False) -> list:
        inputs = inputs.view((-1, )+self.dim_in) # just making sure
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes)

        hybridisation_levels = []
        noises = []
        for l in range(self.max_hybridisation_level+1):
            noise_l = self.sample_noise_controlled(codes, level=l) #size BxD_z
            hybridisation_levels.append(l)
            noises.append(noise_l)

        output = self.decode(torch.vstack(noises), activate)
        return  [output, noises, codes, hybridisation_levels]

    @staticmethod
    def reconstruction_hybrid(X, X_hat, Z, Z_hybrid, level:int):
        """Computes reconstruction loss for hybridised samples
        #TODO: make fancier"""
        N = X_hat.shape[0]
        if level==0: factors= torch.ones(N)
        else: factors = 1/(level*torch.norm(Z-Z_hybrid,1, dim=1)) #TODO: normalise dimension-wise to allow dimensions to change scale during training
        MSEs = torch.sum(F.mse_loss(X_hat, X, reduction="none"),
                         tuple(range(X_hat.dim()))[1:])
        total_cost = torch.matmul(MSEs, factors)/N
        return total_cost

    @staticmethod
    def compute_MMD(Z_all_h:Tensor, Z_max_h:Tensor, kernel="RBF", **kwargs):
        """Naive implementation of MMD in the latent space
        Available kernels: RBF, IMQ, cat (stands for categorical) -- se utils for more ingo
        """
        #TODO: normalise the two vectors per dimension
        #TODO: add switch to subsample first Z (in order to have same dimensionality)
        if kernel=="RBF":
            MMD = utils.MMD(*utils.RBF_kernel(Z_all_h, Z_max_h))
        elif kernel=="IMQ":
            MMD = utils.MMD(*utils.IMQ_kernel(Z_all_h, Z_max_h))
        elif kernel=="cat":
            MMD = utils.MMD(*utils.Categorical_kernel(Z_all_h, Z_max_h, kwargs.get("strict",True), kwargs.get("hierarchy",True)))
        else: raise NotImplementedError("Specified kernel for MMD '"+kernel+"' not implemented.")
        return MMD

    def loss_function(self, *args, **kwargs):
        X_hats = args[0] #(BxM)xD
        Z = args[1] # BxD_z
        Z_hybrid = args[2] #list of tensors of size BxD_z
        H_levels = args[3] #list of hybridisation level used to produce H_hats and Z_hybrid
        X = kwargs["X"]
        B = X.shape[0] #batch size
        L_rec_tot = 0
        L_reg = 0
        for l,level in enumerate(H_levels):
            X_hat = X_hats[l*B:(l+1)*B]
            Z_h = Z_hybrid[l]
            L_rec_tot += self.reconstruction_hybrid(X, X_hat, Z, Z_h, level)
        l_max = np.argmax(H_levels)
        Z_max_h = Z_hybrid[l_max]
        Z_all = torch.vstack(Z_hybrid)
        L_reg = self.compute_MMD(Z_all,Z_max_h)
        #TODO: see how to efficiently compute MMD
        L_rec_tot /= len(H_levels)
        return L_rec_tot, L_reg
