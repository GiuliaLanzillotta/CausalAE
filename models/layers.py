# Implementation of useful layers
import torch
from torch import nn
from torch import Tensor
import numpy as np
from .utils import Mish, act_switch, norm_switch

class FCResidualBlock(nn.Module):
    """ Implements residual fully connected block with adaptive average pooling """
    def __init__(self, dim_in, sizes_lists, act):
        super().__init__()
        self.affine_modules = nn.ModuleList([])
        self.avg_pool_modules = nn.ModuleList([])
        self.act = act()
        prev = dim_in
        for size in sizes_lists:
            self.affine_modules.append(nn.Linear(prev, size))
            self.avg_pool_modules.append(nn.AdaptiveAvgPool1d(size))
            prev = size

    def forward(self, inputs:Tensor) -> Tensor:
        x = inputs
        for affine,pool in zip(self.affine_modules, self.avg_pool_modules):
            delta = self.act(affine(x))
            x = pool(x.unsqueeze(1)).squeeze(1) + delta
        return x

class FCBlock(nn.Module):
    """ Factors the logic for a standard Fully Connected block"""
    def __init__(self, dim_in, sizes_lists, act):
        super().__init__()
        modules = []
        prev = dim_in
        for size in sizes_lists:
            modules.append(nn.Linear(prev, size))
            modules.append(act())
            prev = size
        self.fc = nn.Sequential(*modules)

    def forward(self, inputs:Tensor) -> Tensor:
        return self.fc(inputs)


class ConvBlock(nn.Module):
    """ Implements logic for a convolutional block [conv2d, normalization, activation]"""

    def __init__(self, C_IN, C_OUT, k, s, p, num_groups,
                 pool:bool=False, act=Mish, norm="batch"):
        super(ConvBlock, self).__init__()
        norm_layer = norm_switch(norm, C_OUT, num_groups)
        modules = [nn.Conv2d(C_IN, out_channels=C_OUT, kernel_size=k, stride=s, padding=p),
                   act(),
                   norm_layer]
        if pool: modules.insert(1, nn.MaxPool2d(2,2,0))
        self.block = nn.Sequential(*modules)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)
        return outputs

    @staticmethod
    def compute_output_size(H_in, H_w, s, p):
        #TODO also consider dilation
        return int(np.floor((H_in-H_w+2*p)/s + 1))

    @staticmethod
    def get_filter_size(H_in, s, H_out):
        """Computes filter size to be used to obtain desired output size
         starting with padding 0 and gradually incrementing it."""
        p = 0
        opt =  (H_in +2*p - (H_out - 1)*s)
        while opt+p>H_in or opt<=0:
            p+=1
            opt = (H_in +2*p - (H_out - 1)*s)
        return opt, p

    @staticmethod
    def same_padding(H_in, H_w, s):
        if H_in<H_w: return (H_w-H_in)*s
        return int(((s-1)*H_in + H_w -s)/2)


class PoolingConvBlock(nn.Module):
    """ Specific version of the conv block with same padding and optionally pooling.
    This version differs from the standard conv block in that the dimensionality
    reduction is obtained only through the optional pooling layer."""
    def __init__(self, C_IN, C_OUT, H_in, k, s, num_groups, pool:bool=False, act=Mish, norm="group"):
        super(PoolingConvBlock, self).__init__()
        same_padding = ConvBlock.same_padding(H_in, k, s)
        self.block = ConvBlock(C_IN, C_OUT, k, s, same_padding, num_groups, pool, act=act, norm=norm)

    def forward(self, inputs:Tensor):
        out_conv = self.block(inputs)
        return out_conv


class ResidualConvBlock(PoolingConvBlock):
    """ Residual version of the PoolingConvBlock
    Note: pooling layers addition to obtain dimensionality reduction """
    def __init__(self, C_IN, H_in, k, s, num_groups, pool:bool=False, act=Mish, norm="batch"):
        super(ResidualConvBlock, self).__init__(C_IN, C_IN, H_in, k, s, num_groups,
                                                pool=pool, act=act, norm=norm)

    def forward(self, inputs:Tensor):
        for i, lyr in enumerate(self.block.modules()):
            if i<2: pass #skip first two modules, which are the "containers" of the layers
            elif i==2: out = lyr(inputs) + inputs
            else: out = lyr(out)
        return out

#todo: new version of residual block


class ConvNet(nn.Module):
    """ Implements convolutional net with multiple convolutional 
    blocks  + final flattening """

    def __init__(self, dim_in, final_dim, depth:int, pool_every:int, **kwargs):
        super(ConvNet, self).__init__()
        C, H, W = dim_in
        self.depth = depth
        residual = kwargs.get("residual")
        norm = kwargs.get("norm")
        act = act_switch(kwargs.get("act"))
        channels = kwargs.get("channels")
        # Stacking the conv layers
        modules = []
        h = H # tracking image size
        k = 5 # kernel size
        c = channels
        for l in range(depth):
            # conv block with kernel size 2, size 2 and padding 1 
            # halving the input dimension at every step
            if l==depth-1: c = 1 # reducing depth at the end
            reduce=l%pool_every==0
            if l==1: k=3 # only first layer with kernel=5
            modules.append(PoolingConvBlock(C, c, h, k, 1, 8, pool=reduce, act=act, norm=norm) if (not residual) or (l in [0,depth-1]) # changing num channels here
                           else ResidualConvBlock(C, h, k, 1, 8, pool=reduce, act=act, norm=norm))
            h = h//2 if reduce else h
            k = min(h,k)
            C = c

        # calculating shape of the image after convolution
        self.final_shape = C, h, h # assuming square input image
        assert all(v>0 for v in self.final_shape), "Input not big enough for the convolutions requested"
        flat_dim = int(np.product(self.final_shape))
        modules.append(nn.Flatten())
        if flat_dim!=self.final_shape: # re-adjusting the shape
            modules.append(FCBlock(flat_dim, [final_dim], Mish))
        self.net = nn.Sequential(*modules)


    def forward(self, inputs: Tensor) -> Tensor: 
        outputs = self.net(inputs)
        return outputs

class TransConvBlock(nn.Module):
    """ Implements logic for a transpose convolutional block
    [transposeConv2d, normalization, activation]"""

    def __init__(self, C_IN, C_OUT, k, s, p, num_groups):
        #TODO: add check for the norm
        #TODO: add selector for activation 
        super(TransConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose2d(C_IN, out_channels=C_OUT, kernel_size=k, stride=s, padding=p),
            nn.BatchNorm2d(C_OUT),
            nn.LeakyReLU())

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.block(inputs)
        return outputs


    @staticmethod
    def compute_output_size(H_in, H_w, s, p):
        #TODO also consider dilation
        return int((H_in-1)*s - 2*p + H_w)

class TransConvNet(nn.Module):
    """Implements a Transpose convolutional network (initial reshaping + transpose convolution)"""

    def __init__(self, dim_in, initial_shape, final_shape, depth):
        """
            - initial_shape:(C,H,W) -> shape of the input image to the transpose 
                convolution block
        """
        super(TransConvNet, self).__init__()
        self.initial_shape = initial_shape
        self.final_shape = final_shape
        flat_dim = int(np.product(initial_shape))
        self.fc_reshape = nn.Linear(dim_in, flat_dim)
        self.depth = depth
        C,H,W = initial_shape
        # Stacking the trans-conv layers 
        modules = []
        h = H
        for l in range(depth):
            # transpose conv block with kernel size 2, size 2 and padding 1 
            # doubling the input dimension at every step 
            modules.append(TransConvBlock(C, 64, 5 if l==l-1 else 3, 1, 0, 8))
            h = TransConvBlock.compute_output_size(h, 5 if l==l-1 else 3, 1, 0)
            C = 64
        # reshaping into initial size
        # if the current image size is too small we need to add new transpose convolution blocks
        while h < final_shape[1]:
            self.depth +=1
            modules.append(TransConvBlock(C, 64, 3, 1, 0, 8))
            h = TransConvBlock.compute_output_size(h, 3, 1, 0)
        k, p = ConvBlock.get_filter_size(h, 1, final_shape[1])
        modules.append(ConvBlock(C, final_shape[0], k, 1, p, 1))
        self.trans_net = nn.Sequential(*modules)


    def forward(self, inputs: Tensor) -> Tensor: 
        reshaped = self.fc_reshape(inputs).view((-1,) + self.initial_shape)
        outputs = self.trans_net(reshaped)
        return outputs

class GaussianLayer(nn.Module):
    """Stochastic layer with Gaussian distribution.
    This layer parametrizes a Gaussian distribution on the latent space of size
    "latent_size" """
    def __init__(self, hidden_size, latent_size):
        super().__init__()
        self.latent_size = latent_size
        self.mu = nn.Linear(hidden_size, latent_size)
        self.logvar = nn.Linear(hidden_size, latent_size)

    def forward(self, inputs):
        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.mu(inputs)
        logvar = self.logvar(inputs)

        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        return std * eps + mu, logvar, mu

    def sample_standard(self, num_samples:int) -> Tensor:
        """ Sampling noise from the latent space and generating images
            through the decoder"""
        z = torch.randn(num_samples, self.latent_size)
        return z

class HybridLayer(nn.Module):
    """Stochastic layer based on Hybrid sampling"""
    def __init__(self, dim, unit_dim, N):
        super(HybridLayer, self).__init__()
        self.dim = dim
        self.unit_dim = unit_dim
        self.N = N
        self.weights = torch.ones(self.N) # unnormalised probability weights for each sample (equivalent to uniform)
        self.weights.requires_grad = False
        self.prior = None

    def initialise_prior(self, latent_vectors):
        # randomly selecting latent vectors for the prior
        input_size = latent_vectors.detach().shape[0]
        idx = torch.randperm(input_size).to(latent_vectors.device)
        if input_size<self.N:
            selected_idx = idx[:input_size]
        else: selected_idx = idx[:self.N]
        self.prior = torch.index_select(latent_vectors, 0, selected_idx)

    def update_prior(self, latent_vectors):
        # TODO: make initialisation incremental (store an incremental pool
        #  over different batches and sample from that)
        pass

    def sample_from_prior(self, input_shape):
        # splitting the prior latent vectors into chunks (one for each noise dimension)
        if self.prior is None: raise ValueError("No samples from the prior have been obtained yet")
        num_samples = input_shape[0]
        prior_chunks = torch.split(self.prior, self.unit_dim, dim=1)
        # randomising the order of each chunk
        new_vectors = []
        for chunk in prior_chunks:
            # num_samples x N one-hot matrix
            idx = torch.multinomial(self.weights[:self.prior.shape[0]], num_samples, replacement=True).to(chunk.device)
            new_vectors.append(torch.index_select(chunk, 0, idx))
        noise = torch.cat(new_vectors, dim=1)
        return noise

    def forward(self, inputs):
        """Performs hybrid sampling on the latent space"""
        self.initialise_prior(inputs)
        output = self.sample_from_prior(inputs.shape)
        return output

class AdaIN(nn.Module):
    #TODO: implement AdaIN layer (to be used instead of StrTrf in baselines)
    pass

class StrTrf(nn.Module):
    """Implements the structural transform layer. To be used in SAE."""
    def __init__(self, noise_size, n_features, act=Mish):
        super().__init__()
        self.fc = FCBlock(noise_size, [noise_size*10, noise_size*20, n_features*2], act)
        self.n_features = n_features

    def forward(self, x, z):
        """ z is the noise obtained via hybrid or parametric sampling.
        x is the set of all causal variables computed so far. Thus this is
        the layer where the causal variables interact with the noise terms to generate
        the children variables."""
        params = self.fc(z)
        sigma_z, mu_z = torch.split(params, self.n_features, dim=1)
        # mu_z and sigma_z are one-dimensional vectors: we need to expand their dimensions
        mu_z = mu_z.view(*mu_z.shape, *(1,)*(len(x.shape) - len(mu_z.shape)))
        sigma_z = sigma_z.view(*sigma_z.shape, *(1,) * (len(x.shape) - len(sigma_z.shape)))
        y = mu_z + sigma_z*x
        return y


class UpsampledConv(nn.Module):
    """Implements layer to be used in structural decoder:
    bilinear upsampling + convolution (with activtion) (with no dimensionality reduction)"""
    def __init__(self, upsampling_factor, dim_in, channels:int, filter_size:int=2, stride:int=2,
                 num_groups:int=8, residual:bool=False, act=Mish, norm="batch"):
        super().__init__()
        C, H, W = dim_in
        self.upsampling = nn.Upsample(scale_factor=upsampling_factor, mode="bilinear", align_corners=False)
        padding = ConvBlock.same_padding(H*upsampling_factor, filter_size, stride)
        self.conv_layer = ConvBlock(C, channels, filter_size, stride, padding, num_groups, pool=False, act=act, norm=norm) if not residual \
            else ResidualConvBlock(C, H*upsampling_factor, filter_size, stride, num_groups, pool=False, act=act, norm=norm)
        self.act = act()

    def forward(self, inputs):
        upsmpld = self.upsampling(inputs)
        output = self.act(self.conv_layer(upsmpld))
        return output


class UpsampledConvNet(nn.Module):
    """ Implements upsampling convolutional network, consisting in a series of
    UpsampleConv blocks - to be used in the decoders of fully convolutional networks."""
    def __init__(self, initial_shape, final_shape, depth:int, **kwargs):
        """
        @latent_size:int = size of the noise vector in input (obtained by hybrid/parametric sampling)
        @unit_dim:int = dimension of one "noise unit"- not necessarily 1
        # the noise will be processed one unit at a time -> the noise vector is
        # split in units during the forward pass

        """
        super().__init__()
        self.initial_shape = initial_shape
        self.final_shape = final_shape
        self.depth = depth
        C, H, W = initial_shape # the initial shape refers to the constant input block
        _modules = []
        h = H
        pool_every = kwargs.get("pool_every")
        residual = kwargs.get("residual")
        channels = kwargs.get("channels")
        c = channels # output channels
        norm = kwargs.get("norm")
        for l in range(depth):
            if l == depth-1: c = final_shape[0] # reducing number of channels at the end
            dim_in = (C, h, h)
            _modules.append(UpsampledConv(2 if l%pool_every==0 else 1, dim_in, c, 3, 1, 8,
                                                   residual=residual and C==c, act=Mish, norm=norm))
            h *= 2 if l%pool_every==0 else 1
            C = c
        _modules.append(nn.AdaptiveAvgPool2d(final_shape[1]))
        self.net = nn.Sequential(*_modules)

    def forward(self, inputs):
        """ Simply the forward pass """
        reshaped = inputs.view((-1,) + self.initial_shape)
        outputs = self.net(reshaped)
        return outputs


class SCMDecoder(nn.Module):
    """ Implements SCM layer (to be used in SAE): given a latent noise vector the
    SCM produces the causal variables by ancestral sampling."""
    def __init__(self, initial_shape, final_shape, latent_size, unit_dim, depth:int, **kwargs):
        """
        @latent_size:int = size of the noise vector in input (obtained by hybrid/parametric sampling)
        @unit_dim:int = dimension of one "noise unit"- not necessarily 1
        # the noise will be processed one unit at a time -> the noise vector is
        # split in units during the forward pass

        """
        super().__init__()
        self.final_shape = final_shape
        assert latent_size%unit_dim==0, "The noise vector must be a multiple of the unit dimension"
        self.latent_size = latent_size
        self.unit_dim = unit_dim
        self.depth = depth
        self.hierarchy_depth = latent_size//unit_dim
        assert self.depth>=self.hierarchy_depth, "Given depth for decoder network is not enough for given noise size"
        C, H, W = initial_shape # the initial shape refers to the constant input block
        self.conv_modules = nn.ModuleList([])
        self.str_trf_modules = nn.ModuleList([])
        h = H
        pool_every = kwargs.get("pool_every")
        residual = kwargs.get("residual")
        channels = kwargs.get("channels")
        c = channels # output channels
        norm = kwargs.get("norm")
        act = act_switch(kwargs.get("act"))
        for l in range(depth):
            if l == depth-1: c = final_shape[0] # reducing number of channels at the end
            dim_in = (C, h, h)
            if l<self.hierarchy_depth:
                self.str_trf_modules.append(StrTrf(self.unit_dim, C, act=act))
            self.conv_modules.append(UpsampledConv(2 if l%pool_every==0 else 1, dim_in, c, 3, 1, 8,
                                                   residual=residual and C==c, act=act, norm=norm))
            h *= 2 if l%pool_every==0 else 1
            C = c
        self.shape_adjusting_block = nn.AdaptiveAvgPool2d(final_shape[1])

    def forward(self, x, z, mode="auto"):
        """ Implements forward pass with 2 possible modes:
        - auto: no hybrid sampling (only the convolution)
        - hybrid: hybrid sampling included """
        if mode=="auto":
            for conv in self.conv_modules:
                x = conv(x)
        elif mode=="hybrid":
            for l in range(self.depth):
                if l<self.hierarchy_depth:
                    z_i = z[:,l*self.unit_dim:(l+1)*self.unit_dim]
                    x = self.str_trf_modules[l](x, z_i)
                x = self.conv_modules[l](x)
        else: raise NotImplementedError("Unknown specified forward mode for SCM")
        outputs = self.shape_adjusting_block(x)
        return outputs







