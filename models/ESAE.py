#Code for E(quivariant)SAE experiment
#Experiment number one on SAE independence
from typing import List

import numpy as np
from torch import nn
from torch import Tensor
import torch
from torchvision.transforms import Normalize
from . import ConvNet, SCMDecoder, HybridLayer, FCBlock, FCResidualBlock, GenerativeAE, SAE, utils, VecSCM, VecSCMDecoder, HybridAE
from torch.nn import functional as F
from .utils import act_switch
from abc import ABCMeta, abstractmethod, ABC


class EHybridAE(HybridAE, ABC):
    """Equivariant version of the HybridSAE
    Defines levels of hybrid sampling and integrates hybridisation into forward pass"""
    @abstractmethod
    def __init__(self, params:dict):
        super(EHybridAE, self).__init__(params)

    def sample_noise_controlled(self, latent_vecs: Tensor, level:int):
        noise = self.hybrid_layer.controlled_sampling(latent_vecs, level, use_prior=False)
        return noise

    @abstractmethod
    def decode(self, noise:Tensor, activate:bool):
        raise NotImplementedError

    def controlled_sample_noise_from_prior(self, device:str, num_samples:int):
        dummy_latents = torch.zeros(11,11,11).to(device)
        samples = []
        for level in range(self.max_hybridisation_level+1):
            noise = self.hybrid_layer.controlled_sampling(dummy_latents, level, num_samples, use_prior=True)
            samples.append(noise)
        return samples

    def reconstruction_hybrid(self, X, X_hat, Z, Z_hybrid, level:int, device:str, use_MSE:bool=True):
        """Computes reconstruction loss for hybridised samples
        #NOTSURE - not beautiful"""
        N = X_hat.shape[0]
        if level==0: factors= torch.ones(N).to(device)
        else:
            #FIXME: this per-dimension standardisation disrupts inter units relationships?
            means = Z.mean(dim=0, keepdim=True)
            stds = Z.std(dim=0, keepdim=True)
            ZN = (Z - means) / stds
            ZHN = (Z_hybrid -means) / stds  #standardise both samples with respect to the input distribution
            # in this way the changes to the distribution brought by hybridisation will be more evident
            # (ideally these two distributions should be the same)
            factors = (1/((level*torch.norm(ZN-ZHN,1, dim=1))+10e-5)).to(device) #TODO: change norm here (inlude hierarchy info)
        if use_MSE:
            distance = torch.sum(F.mse_loss(self.act(X_hat), X, reduction="none"),
                         tuple(range(X_hat.dim()))[1:])
        else:
            distance = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                        tuple(range(X_hat.dim()))[1:])
        total_cost = torch.matmul(distance.to(device), factors)/N
        return total_cost

    def reconstruct(self, x: Tensor, activate:bool) -> Tensor:
        """ Simply wrapper to directly obtain the reconstructed image from
        the net"""
        inputs = x.view((-1, )+self.dim_in) # just making sure
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes, integrate=True)
        output = self.decode(codes, activate)
        return  output

    def forward(self, inputs: Tensor, **kwargs) -> List:
        activate = kwargs.get('activate',False)
        integrate = kwargs.get('integrate',True)
        codes = self.encode(inputs)
        self.hybrid_layer.update_prior(codes, integrate=integrate)

        hybridisation_levels = []
        noises = []
        for l in range(self.max_hybridisation_level+1):
            noise_l = self.sample_noise_controlled(codes, level=l) #size BxD_z
            hybridisation_levels.append(l)
            noises.append(noise_l)

        output = self.decode(torch.vstack(noises), activate)
        return  [output, codes, noises, hybridisation_levels]

    def loss_function(self, *args, **kwargs):
        X_hats = args[0] #(BxM)xD
        Z = args[1] # BxD_z
        Z_hybrid = args[2] #list of tensors of size BxD_z
        H_levels = args[3] #list of hybridisation level used to produce H_hats and Z_hybrid
        X = kwargs["X"]
        lamda = kwargs.get('lamda')
        device = kwargs.get('device','cpu')
        use_MSE = kwargs.get('use_MSE',True)
        B = X.shape[0] #batch size
        L_rec_tot = 0
        for l,level in enumerate(H_levels):
            X_hat = X_hats[l*B:(l+1)*B]
            Z_h = Z_hybrid[l]
            L_rec_tot += self.reconstruction_hybrid(X, X_hat, Z, Z_h, level, device=device, use_MSE=use_MSE)
        l_max = np.argmax(H_levels)
        Z_max_h = Z_hybrid[l_max]
        Z_all = torch.vstack(Z_hybrid)
        Z_all = Z_all[torch.randperm(Z_all.shape[0])[:B]] #B should also be the number of vectors in Z_max
        L_reg = utils.compute_MMD(Z_all,Z_max_h, kernel=self.kernel_type, **self.params, device=device)
        L_rec_tot /= len(H_levels)
        loss = L_rec_tot + lamda*L_reg
        return{'loss': loss, 'Reconstruction_loss':L_rec_tot, 'Regularization_loss':L_reg}



class ESAE(EHybridAE, SAE):
    """Equivariant version of the SAE"""
    def __init__(self, params:dict) -> None:
        super(ESAE, self).__init__(params)
        self.max_hybridisation_level = self.latent_size//self.unit_dim
        # M = number of hybrid samples to be drawn during forward pass
        # note that the minimum number of hybrid samoles coincides with the
        # number of hybridisation levels
        self.kernel_type = params["MMD_kernel"]
        #TODO: allow M
        #self.M = max(params["num_hybrid_samples"], self.latent_units)

    def decode(self, noise:Tensor, activate:bool):
        return SAE.decode(self, noise, activate)


class VecESAE(EHybridAE):
    """Version of ESAE model for vector based (not image based) data"""
    def __init__(self, params: dict, dim_in: int, full: bool, **kwargs) -> None:
        """ full: whether to use the VecSCMDecoder layer as a decoder"""
        super(VecESAE, self).__init__(params)
        self.dim_in = dim_in[0]
        self.max_hybridisation_level = self.latent_size//self.unit_dim
        self.kernel_type = params["MMD_kernel"]

        # dim_in is a single number (since the input is a vector)
        layers = list(torch.linspace(self.dim_in, self.latent_size, steps=params["depth"]).int().numpy())
        self.encoder = FCBlock(self.dim_in, layers, act_switch(params.get("act")))
        self.full = full
        if not full:
            scm = VecSCM(self.latent_size, self.unit_dim, act=params.get("act"))
            reverse_encoder = FCBlock(self.latent_size, reversed(layers), act_switch(params.get("act")))
            self.decoder = nn.Sequential(scm, reverse_encoder)
        else: self.decoder = VecSCMDecoder(self.latent_size, self.unit_dim, list(reversed(layers)), act=params.get("act"))


    def decode(self, noise:Tensor, activate:bool):
        # since x is a constant we're always going to get the same output
        if not self.full:
            output = self.decoder(noise)
        else:
            x = torch.ones_like(noise).to(noise.device)
            output = self.decoder(x, noise)
        if activate: output = self.act(output)
        return output