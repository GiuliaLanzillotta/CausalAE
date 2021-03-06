"""Implementation of various utilities """
import math

from torch import nn
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Gumbel
from torchvision.transforms import Normalize


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
    if name=="elu":return nn.ELU
    raise  NotImplementedError(f"Specified activation -{name}- not supported")


def standard_initialisation(m, non_linearity='leaky_relu'):
    """ More info on weight init in Pytorch:
    https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html
    https://pytorch.org/docs/stable/nn.init.html
    https://www.deeplearning.ai/ai-notes/initialization/
    """
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Parameter)):
        if non_linearity in ['relu','leaky_relu']:
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity=non_linearity)
        else: nn.init.xavier_normal_(m.weight)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def Gaussian_kernel(X:Tensor, sigma:float):
    """Computes Gaussian kernel similarity of data X (to be used to estimate Renyis entropy from data)
    X has dimensionality (N x d), where N is the number of samples"""
    N, D = X.shape
    norms = torch.sum(X**2, axis=1, keepdim=True)
    dotprods = torch.matmul(X, X.T)
    distances = norms + norms.T - 2. * dotprods # N x N matrix
    res = torch.exp(-distances/(4*sigma**2))
    C = torch.pow(2 * math.pi * sigma * math.sqrt(2), (-D / 2)) # multiplicative constant
    res = C*res
    return res



def RBF_kernel(X:Tensor, Y:Tensor, device:str):
    """
    Computes similarity as MMD between X and Y according to RBF kernel, using sigma equal to the median distance between
    vectors computed wrt to both samples
    """
    norms_pz = torch.sum(X**2, axis=1, keepdim=True)
    dotprods_pz = torch.matmul(X, X.T)
    distances_pz = norms_pz + norms_pz.T - 2. * dotprods_pz

    norms_qz = torch.sum(Y**2, axis=1, keepdim=True)
    dotprods_qz = torch.matmul(Y, Y.T)
    distances_qz = norms_qz + norms_qz.T - 2. * dotprods_qz

    dotprods = torch.matmul(X, Y.T)
    distances = norms_pz + norms_qz.T - 2. * dotprods

    # Median heuristic for the sigma^2 of Gaussian kernel
    sigma2_k = torch.quantile(distances_pz, q = 0.5)
    sigma2_k += torch.quantile(distances_qz, q = 0.5)


    res1 = torch.exp( - distances_pz / 2. / sigma2_k)
    #res1 = torch.multiply(res1, 1. - torch.eye(N).to(device)) #this is wrong

    res2 = torch.exp( - distances_qz / 2. / sigma2_k)
    #res2 = torch.multiply(res2, 1. - torch.eye(M).to(device))

    res3 = torch.exp( - distances / 2. / sigma2_k)

    return res1,res2,res3

def IMQ_kernel(X:Tensor, Y:Tensor, device:str):
    """Inverse multiquadratic kernels- less sensible to outliers
    Formula: k(x, y) = C / (C + ||x - y||^2)
    credits: @https://github.com/tolstikhin/wae/blob/master/wae.py
            @https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_mmd.py
            @https://arxiv.org/pdf/1711.01558.pdf
    """
    M, D = X.shape; N, _ = Y.shape
    norms_x = X.pow(2).sum(1, keepdim=True)  # M x 1
    prods_x = torch.mm(X, X.t())  # M x M
    dists_x = norms_x + norms_x.t() - 2 * prods_x

    norms_y = Y.pow(2).sum(1, keepdim=True)  # N x 1
    prods_y = torch.mm(Y, Y.t())  # N x N
    dists_y = norms_y + norms_y.t() - 2 * prods_y

    dot_prd = torch.mm(X, Y.t())
    dists_c = norms_x + norms_y.t() - 2 * dot_prd

    stats = 0
    for scale in [.1, .2, .5, 1., 2., 5., 10.]:
        C = 2 * D * 1.0 * scale
        res1 = C / (C + dists_x)
        res2 = C / (C + dists_y)

        res1 = (1 - torch.eye(M, device=device)) * res1
        res2 = (1 - torch.eye(N, device=device)) * res2

        res1 = res1.sum() / (M * (M - 1))
        res2 = res2.sum() / (N * (N - 1))

        res3 = C / (C + dists_c)
        res3 = res3.sum() * 2. / (M**2)
        stats += res1 + res2 - res3

    return stats


def Categorical_kernel(X,Y, device, strict=False, hierarchy=False):
    """
    Computes similarity between X and Y according to a categorical kernel
    See here for inspo: https://upcommons.upc.edu/bitstream/handle/2117/23347/KernelCATEG_CCIA2013.pdf?sequence=1
    strict: whether to use L1 or L_0 norm for the distance between any two latent vectors
    """
    #NOTSURE: check whether L0 and L1 norm actually make up a valid kernel (PD)

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
        hierarchy_weights = torch.linspace(0,1,Dz+2)[1:-1].to(device)
        hierarchy_weights = inverting_fun(hierarchy_weights)
    else:
        hierarchy_weights = torch.ones(Dz).to(device)

    if not strict:
        delta_X = torch.max(X,axis=0)[0] - torch.min(X,axis=0)[0]
        delta_Y = torch.max(Y,axis=0)[0] - torch.min(Y,axis=0)[0]

    similarities1 = torch.zeros(N,N, dtype=torch.float).to(device)
    for i in range(N):
        #TODO: check this delta - is this the only way to obtain a similarity from the distance?
        similarities_i = delta_X - torch.abs(X-X[i]) if not strict else (X==X[i]).type(torch.float)
        similarities1[:,i] = torch.matmul(similarities_i,hierarchy_weights)

    similarities2 = torch.zeros(M,M, dtype=torch.float).to(device)
    similarities3 = torch.zeros(N,M, dtype=torch.float).to(device)
    for i in range(M):
        similarities_i = delta_Y - torch.abs(Y-Y[i]) if not strict else (Y==Y[i]).type(torch.float)
        similarities2[:,i] = torch.matmul(similarities_i,hierarchy_weights)
        similarities_i = (delta_X + delta_Y) - torch.abs(X-Y[i]) if not strict else (X==Y[i]).type(torch.float)
        similarities3[:,i] = torch.matmul(similarities_i,hierarchy_weights)

    return similarities1,similarities2,similarities3

def MMD(similaritiesPP, similaritiesQQ, similaritiesPQ):
    """ Computes MMD given the three matrices of distances given as input
    (obtained through one of the three above functions)"""
    N = similaritiesPP.shape[0]
    M = similaritiesQQ.shape[0]
    res1 = torch.sum(similaritiesPP) / float(N * (N - 1))
    res2 = torch.sum(similaritiesQQ) / float(M * (M - 1))
    res3 = torch.sum(similaritiesPQ) * 2. / float(N * M)

    return res1 + res2 - res3

def compute_MMD(fromP:Tensor, fromQ:Tensor, kernel="RBF", **kwargs):
    """Naive implementation of MMD in the latent space
    Available kernels: RBF, IMQ, cat (stands for categorical) -- se utils module for more info
    """

    # normalising to avoid implicitly penalising for magnitude with kernels that are
    # sensitive to the spread of the variable - e.g. if we don't normalise then we get
    # latents with more spread to be penalised more than the others by the regularisation
    # term -> which will in turn make them react stronger to it
    # the idea is to keep the natural spread that the latent distribution has

    standardise = kwargs.get('standardise',False)
    if standardise:
        all_ = torch.vstack([fromP,fromQ])
        means = all_.mean(dim=0, keepdim=True)
        stds = all_.std(dim=0, keepdim=True)
        PN = (fromP - means) / stds
        QN = (fromQ -means) / stds  #standardise both samples with respect to the input distribution
    else: PN = fromP; QN = fromQ

    device = kwargs.get('device')
    #TODO: insert hierarchy RBF -why though? we would end up giving more slack to some dimensions: is this what we want?
    if kernel=="RBF":
        _MMD = MMD(*RBF_kernel(PN, QN, device))
    elif kernel=="IMQ":
        _MMD = IMQ_kernel(PN, QN, device)
    elif kernel=="cat":
        _MMD = MMD(*Categorical_kernel(PN, QN, device, kwargs.get("strict",True), kwargs.get("hierarchy",True)))
    else: raise NotImplementedError("Specified kernel for MMD '"+kernel+"' not implemented.")

    return _MMD

def pixel_losses(X, X_hat, act):
    """ Computes both MSE and BCE loss for X and X_hat"""
    # mean over batch of the sum over all other dimensions
    MSE = torch.sum(F.mse_loss(act(X_hat), X, reduction="none"),
                    tuple(range(X_hat.dim()))[1:]).mean()
    BCE = torch.sum(F.binary_cross_entropy_with_logits(X_hat, X, reduction="none"),
                    tuple(range(X_hat.dim()))[1:]).mean()
    return MSE,BCE

def KL_multiple_univariate_gaussians(mus_1, mus_2, logvars_1, logvars_2, reduce=False):
    """KL divergence between multiple tuples of univariate guassians:
    all input have size mxD"""
    KL = (logvars_2 - logvars_1) - 0.5
    numr = logvars_1.exp().pow(2) + (mus_1 - mus_2).pow(2)
    denm = 2*(logvars_2.exp().pow(2))
    KL += torch.div(numr,denm)
    if reduce: KL = torch.sum(KL, dim=1).mean() # sum over D -> m x 1 -- mean over m -> 1x1
    return KL

def distribution_parameter_distance(mus_1, mus_2, logvars_1, logvars_2, reduce=False, ignore_variance=False):
    """Measures distance between two distributions parametrised by mean and variance as
    the distance of their parameters"""
    distance = (mus_1 - mus_2).abs()
    if not ignore_variance:
        distance += (logvars_1 - logvars_2).abs()
    if reduce: distance = torch.sum(distance, dim=1).mean()
    return distance

def gumbel_trick(logits:Tensor, tau:float, device:str):
    """
    logits: dx1 weigth vector parametrising the log probabilities of sampling from a Bernoulli
    tau:temperature parameter for the softmax
    Returns: a dx1 tensor with values sampled according to the gumbel-softmax distribution on the logits"""
    with torch.no_grad: gumbel_noise = Gumbel(0,1).sample(logits.shape).to(device)
    noised_logits = (logits + gumbel_noise)/tau
    return noised_logits

def continuous_DAG_constraint(M, power):
    """Implementation of the continuous differential DAG constraint on the input matrix
    as shown in https://arxiv.org/abs/1904.10098.
    Code from https://github.com/fishmoon1234/DAG-GNN/blob/master/src/train.py """
    M_square = torch.eye(power).double().to(M.device) + torch.div(M*M, power)
    M_power = torch.matrix_power(M_square, power)
    H = torch.trace(M_power) - power
    return H

if __name__ == '__main__':
    #TESTING
    X = torch.randn(100,10)
    Y = torch.randn(100,10)
    res1,res2,res3 = RBF_kernel(X,Y, device=X.device)





