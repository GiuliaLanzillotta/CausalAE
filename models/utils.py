"""Implementation of various utilities """
import math

from torch import nn
import torch
import torch.nn.functional as F
from torch import Tensor


class Mish(nn.Module):
    """ MISH nonlinearity - https://arxiv.org/abs/1908.08681"""
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def norm_switch(name:str, channels:int, num_groups:int=0, shape=None):
    if name=="batch":return nn.BatchNorm2d(channels)
    if name=="group":return nn.GroupNorm(min(num_groups, channels), channels)
    if name=="layer":return nn.LayerNorm(shape)
    raise NotImplementedError(f"Specified norm -{name}- not supported")

def act_switch(name:str):
    if name=="mish": return Mish
    if name=="relu": return nn.ReLU
    if name=="leaky_relu":return nn.LeakyReLU
    raise  NotImplementedError(f"Specified activation -{name}- not supported")


def standard_initialisation(m, non_linearity='leaky_relu'):
    """ More info on weight init in Pytorch:
    https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    https://pytorch.org/docs/stable/nn.init.html
    https://www.deeplearning.ai/ai-notes/initialization/
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if non_linearity in ['relu','leaky_relu']:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=non_linearity)
        else: nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def RBF_kernel(X:Tensor, Y:Tensor):
    """
    Computes similarity between X and Y according to RBF kernel, using sigma equal to the median distance between
    vectors computed wrt to both samples
    """
    N = X.shape[0]
    M = Y.shape[0]

    norms_pz = torch.sum(X**2, axis=1, keepdim=True)
    dotprods_pz = torch.matmul(X, X.T)
    distances_pz = norms_pz + norms_pz.T - 2. * dotprods_pz

    norms_qz = torch.sum(Y**2, axis=1, keepdim=True)
    dotprods_qz = torch.matmul(Y, Y.T)
    distances_qz = norms_qz + norms_qz.T - 2. * dotprods_qz

    dotprods = torch.matmul(X, Y.T)
    distances = norms_qz + norms_pz.T - 2. * dotprods

    sigma2_k = torch.quantile(distances_pz, q = 0.5)
    sigma2_k += torch.quantile(distances_qz, q = 0.5)

    res1 = torch.exp( - distances_qz / 2. / sigma2_k)
    res1 = torch.multiply(res1, 1. - torch.eye(N))
    res1 = torch.sum(res1) / float(N * (N - 1))

    res2 = torch.exp( - distances_pz / 2. / sigma2_k)
    res2 = torch.multiply(res2, 1. - torch.eye(M))
    res2 = torch.sum(res2) / float(M * (M - 1))

    res3 = torch.exp( - distances / 2. / sigma2_k)
    res3 = torch.sum(res3) * 2. / float(N * M)
    stat = res1 + res2 - res3

    return stat

def IMQ_kernel(X:Tensor, Y:Tensor):
    """Inverse multiquadratic kernels- less sensible to outliers
    Formula: k(x, y) = C / (C + ||x - y||^2)
    credits: @https://github.com/tolstikhin/wae/blob/master/wae.py
            @https://arxiv.org/pdf/1711.01558.pdf
    """
    N = X.shape[0]
    M = Y.shape[0]

    norms_pz = torch.sum(X**2, axis=1, keepdim=True)
    dotprods_pz = torch.matmul(X, X.T)
    distances_pz = norms_pz + norms_pz.T - 2. * dotprods_pz

    norms_qz = torch.sum(Y**2, axis=1, keepdim=True)
    dotprods_qz = torch.matmul(Y, Y.T)
    distances_qz = norms_qz + norms_qz.T - 2. * dotprods_qz

    dotprods = torch.matmul(X, Y.T)
    distances = norms_qz + norms_pz.T - 2. * dotprods

    # expected squared distance between two multivariate Gaussian vectors drawn from prior
    Cbase = 2.*float(X.shape[1])*torch.mean(distances_qz)

    stat = 0.
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = Cbase * scale
        res1 = C / (C + distances_qz)
        res1 = torch.multiply(res1, 1. - torch.eye(N))
        res1 = torch.sum(res1) / float(N * (N - 1))

        res2 = C / (C + distances_pz)
        res2 = torch.multiply(res2, 1. - torch.eye(M))
        res2 = torch.sum(res2) / float(M * (M - 1))

        res3 = C / (C + distances)
        res3 = torch.sum(res3) * 2. / float(N * M)
        stat += res1 + res2 - res3

    return stat

def Categorical_kernel(X,Y, strict=False, hierarchy=False):
    """
    Computes similarity between X and Y according to a categorical kernel
    See here for inspo: https://upcommons.upc.edu/bitstream/handle/2117/23347/KernelCATEG_CCIA2013.pdf?sequence=1
    strict: whether to use L1 or L_0 norm for the distance between any two latent vectors
    """
    N = X.shape[0]
    M = Y.shape[0]
    Dz = X.shape[1]
    def inverting_fun(z, alpha=2.0):
        """assumes z \in [0,1]"""
        s = torch.pow((1. - torch.pow(z,alpha)),(1/alpha))
        return s

    #basically the idea is to reformulate the Overlap kernel exploiting the structure
    # k_u (zi_zj) = h_alpha(hierarchy_level)*||z_i-z_j||  -univariate kernel
    # k(z_i,z_j) = \sum_s { k_u(z_is, z_js) } -multivariate kernel : simply sum of scores over multiple dimensions

    #Taking into account the hierarchy means that we're going to penalise more the difference
    #between the two distributions when there is more coherence in-distribution
    #than between-distribution among the higher level features
    if hierarchy:
        hierarchy_weights = torch.linspace(0,1,Dz+2)[1:-1]
        hierarchy_weights = inverting_fun(hierarchy_weights)
    else:
        hierarchy_weights = torch.ones(Dz)

    res1 = 0.
    if not strict:
        delta_X = torch.max(X,axis=0)[0] - torch.min(X,axis=0)[0]
        delta_Y = torch.max(Y,axis=0)[0] - torch.min(Y,axis=0)[0]
    for i in range(N):
        similarities_i = delta_X - torch.abs(X-X[i]) if not strict else (X==X[i]).type(torch.float)
        res1 += torch.sum(torch.matmul(similarities_i,hierarchy_weights))
    res1 /= float(N*(N-1))

    res2 = 0.
    res3 = 0.
    for i in range(M):
        similarities_i = delta_Y - torch.abs(Y-Y[i]) if not strict else (Y==Y[i]).type(torch.float)
        res2 += torch.sum(torch.matmul(similarities_i,hierarchy_weights))
        similarities_i = (delta_X + delta_Y) - torch.abs(X-Y[i]) if not strict else (X==Y[i]).type(torch.float)
        res3 += torch.sum(torch.matmul(similarities_i,hierarchy_weights))
    res2 /= float(M*(M-1))
    res3 /= float(M*N)

    return res1+res2-2*res3





