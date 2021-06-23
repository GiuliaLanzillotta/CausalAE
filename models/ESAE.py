#Code for E(quivariant)SAE experiment
#Experiment number one on SAE independence
import numpy as np
from torch import nn
from torch import Tensor
import torch
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE, SAE
from torch.nn import functional as F
from .utils import act_switch

def RBF_kernel(X, Y):
    """
    Computes similarity between X and Y according to RBF kernel
    """
    #TODO
    half_size = (n * n - n) / 2
    norms_pz = tf.reduce_sum(tf.square(sample_pz), axis=1, keep_dims=True)
    dotprods_pz = tf.matmul(sample_pz, sample_pz, transpose_b=True)
    distances_pz = norms_pz + tf.transpose(norms_pz) - 2. * dotprods_pz

    norms_qz = tf.reduce_sum(tf.square(sample_qz), axis=1, keep_dims=True)
    dotprods_qz = tf.matmul(sample_qz, sample_qz, transpose_b=True)
    distances_qz = norms_qz + tf.transpose(norms_qz) - 2. * dotprods_qz

    dotprods = tf.matmul(sample_qz, sample_pz, transpose_b=True)
    distances = norms_qz + tf.transpose(norms_pz) - 2. * dotprods

    sigma2_k = tf.nn.top_k(tf.reshape(distances, [-1]), half_size).values[half_size - 1]
    sigma2_k += tf.nn.top_k(tf.reshape(distances_qz, [-1]), half_size).values[half_size - 1]

    res1 = tf.exp( - distances_qz / 2. / sigma2_k)
    res1 += tf.exp( - distances_pz / 2. / sigma2_k)
    res1 = tf.multiply(res1, 1. - tf.eye(n))
    res1 = tf.reduce_sum(res1) / (nf * nf - nf)
    res2 = tf.exp( - distances / 2. / sigma2_k)
    res2 = tf.reduce_sum(res2) * 2. / (nf * nf)
    stat = res1 - res2

    return stat

def Categorical_kernel(X,Y):
    """
    Computes similarity between X and Y according to a categorical kernel
    See here for inspo: https://upcommons.upc.edu/bitstream/handle/2117/23347/KernelCATEG_CCIA2013.pdf?sequence=1
    """
    #TODO: take into account hierarchy
    #basically the idea is to reformulate the Overlap kernel exploiting the structure

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
    def compute_MMD(Z_all_h:Tensor, Z_max_h:Tensor, kernel):
        """Naive implementation of MMD in the latent space"""
        MN = Z_all_h.shape[0]
        P = Z_max_h.shape[0]
        L_reg_all, L_reg_max, L_reg_mix = 0,0,0
        for i in range(MN):
            for j in range(MN):
                L_reg_all+= kernel(Z_all_h[i],Z_all_h[j])
            for p in range(P):
                L_reg_mix+= kernel(Z_all_h[i],Z_max_h[p])
        for i in range(P):
            for j in range(P):
                L_reg_max+= kernel(Z_max_h[i],Z_max_h[j])
        L_reg_all /= (MN*(MN-1))
        L_reg_mix /= (MN*P)
        L_reg_max /= (P*(P-1))

        return L_reg_max+L_reg_mix+L_reg_all

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
