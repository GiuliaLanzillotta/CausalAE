""" Implementation of CausalVAE network from the paper CausalVAE: Disentangled Representation Learning via Neural Structural Causal Models"""
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn
from models import VAEBase, GaussianLayer
from .utils import pixel_losses, KL_multiple_univariate_gaussians, continuous_DAG_constraint, act_switch, standard_initialisation
from . import ConvNet, GaussianLayer, GenerativeAE, UpsampledConvNet, FCBlock, DittadiConvNet, DittadiUpsampledConv, \
    VecSCM


class CausalVAE(VAEBase):

    def __init__(self, params: dict) -> None:
        super(CausalVAE, self).__init__(params)
        # being a VAE it automatically has latent_size, unit_dim and gaussian latent
        self.full_latent_size = self.latent_size*self.unit_dim
        self.gaussian_latent = GaussianLayer(self.full_latent_size, self.full_latent_size, params["gaussian_init"])
        # define SCM and mask layers
        self.A = nn.Parameter(torch.Tensor(self.latent_size, self.latent_size), requires_grad=True)
        non_linearity_layers = [self.latent_size, 32, self.unit_dim]
        self.G = nn.ModuleList([FCBlock(self.latent_size*self.unit_dim, non_linearity_layers, act_switch(params["act"])) for i in range(self.latent_size)])
        self.dim_in = params['dim_in'] # C, H, W
        self.DAG_loss_parameters = {
            "lamda":0.5,
            "c":1.0,
            "threshold":0.
        }

        # Building encoder and decoder - two different architectures possible
        self.convolutional = params['convolutional'] # Boolean parameter, deciding whether the encoder/decoder are convolutional structures
        if self.convolutional:
            conv_net = ConvNet(depth=params["enc_depth"], **params)
            fc_enc = FCBlock(conv_net.final_dim, [128, 64, self.latent_size*self.unit_dim], act_switch(params["act"]))
            fc_dec = FCBlock(self.latent_size*self.unit_dim, [64, 128, conv_net.final_dim], act_switch(params["act"]))
            self.encoder = nn.Sequential(conv_net, fc_enc)
            self.decoder_initial_shape = conv_net.final_shape
            deconv_net = UpsampledConvNet(self.decoder_initial_shape, final_shape=self.dim_in, depth=params["dec_depth"], **params)
            self.decoder = nn.Sequential(fc_dec, deconv_net)
        else:
            flat_dim_in = 1
            for d in list(self.dim_in): flat_dim_in*=d
            self.encoder = FCBlock(flat_dim_in, [900,300,self.latent_size*self.unit_dim], act_switch(params["act"]))
            self.decoder = nn.ModuleList([FCBlock(self.unit_dim,[300,300,1024,flat_dim_in],act_switch(params["act"])) for _ in range(self.latent_size)])

        #initialisation
        nn.init.normal_(self.A) #TODO: initialise to constant (0.5) if pretraining present
        self.G.apply(lambda m: standard_initialisation(m, params.get("act")))
        self.encoder.apply(lambda m: standard_initialisation(m, params.get("act")))
        self.decoder.apply(lambda m: standard_initialisation(m, params.get("act")))


    def encode(self, inputs: Tensor, **kwargs):
        if not self.convolutional:
            inputs = torch.flatten(inputs, start_dim=1)
        codes = self.encoder(inputs)
        eps, logvar, mu = self.gaussian_latent(codes)
        return [eps, mu, logvar]

    def decode(self, noise: Tensor, activate:bool) -> Tensor:
        if self.convolutional:
            out = self.decoder(noise)
        else:
            split_noise = torch.split(noise, self.unit_dim, dim=1)
            out = 0
            for i in range(self.latent_size):
                out += self.decoder[i](split_noise[i])
            #Turn back into image format
            out = torch.reshape(out, (-1,)+self.dim_in)
        if activate: out = self.act(out)
        return out

    def encode_mu(self, inputs:Tensor, **kwargs) -> Tensor:
        return self.encode(inputs)[1]

    def DAG_transform(self, eps:Tensor, **kwargs) -> Tensor:
        """Transforms the noise exogenous variables into endogenous ones
        Note: eps has shape (BATCH, N, K) , N being latent size and K being the unit size"""
        #TODO: check args
        C = torch.inverse(torch.eye(self.latent_size).to(eps.device) - self.A.T) # shape N x N
        z = torch.matmul(C, eps) # BATCH, N, K
        return z

    def mask_latents(self, z:Tensor, eps:Tensor, **kwargs) -> Tensor:
        z_hat = torch.zeros_like(z).to(z.device)
        # for each dimension of z apply a different non-linearity
        for i in range(self.latent_size):
            # z is a 3D vector so we need to expand A columns dimensionality
            zi_parents = self.A[:,i].reshape(-1,1)*z # this is where the masking happens
            z_hat[:,i,:] = self.G[i](zi_parents.reshape(-1,self.latent_size*self.unit_dim)) + eps[:,i,:]
        return z_hat

    def sample_from_conditional_prior(self, labels:Tensor, num_samples:int, device:str):
        """Sampling endogenous variables (z) from the conditional p(z|u)
        Note: u has to be provided in vectorised form"""
        return self.gaussian_latent.sample_parametric(num_samples, mus=labels.to(device),
                                                      logvars=torch.ones_like(labels).to(device),
                                                      device=device)

    def sample_noise_from_prior(self, num_samples: int, **kwargs):
        """Sampling exogenous variables (epsilon) from the standard Gaussian prior p(eps)"""
        return self.gaussian_latent.sample_standard(num_samples, device=kwargs['device'])

    def sample_latent(self, num_samples:int, labels:Tensor, **kwargs):
        """Sampling both endogenous and exogenous variables.
        Both epsilon and z are sampled in vectorised form. """
        device = kwargs.get('device',labels.device)
        tot = min(num_samples, labels.shape[0])
        eps = self.sample_noise_from_prior(tot, device=device)
        z = self.sample_from_conditional_prior(labels=labels.repeat_interleave(self.unit_dim, dim=1),
                                               num_samples=tot, device=device)
        return eps, z

    def sample_latent_unconditionally(self, num_samples:int, device:str, **kwargs):
        """Sampling endogenous and exogenous variables unconditionally (without masking).
        """
        eps = self.sample_noise_from_prior(num_samples, device=device)
        eps_MT = eps.reshape(-1, self.latent_size, self.unit_dim)
        z = self.DAG_transform(eps_MT).reshape(-1, self.latent_size*self.unit_dim)
        return eps, z


    def generate(self, num_samples:int, activate:bool, **kwargs) -> Tensor:
        u = kwargs.get('u')
        if u is not None:
            eps, z = self.sample_latent(num_samples, labels=u, **kwargs)
            z_masked_MT = self.mask_latents(z.reshape(-1, self.latent_size, self.unit_dim),
                                     eps.reshape(-1, self.latent_size, self.unit_dim))
            z_fin = z_masked_MT.reshape(-1, self.latent_size*self.unit_dim)
        else:
            # if the labels are not provided we can simply sample a random set from the dataset
            eps, z_fin = self.sample_latent_unconditionally(num_samples, **kwargs)
        output = self.decode(z_fin, activate=activate)
        return output

    def forward_SCM(self, eps: Tensor):
        """Performs the forward pass through DAG and Mask layers, taking
        care of the various reshapings."""
        # turning into matrix form (nxk)
        eps_MT = eps.reshape(-1, self.latent_size, self.unit_dim)
        z_MT = self.DAG_transform(eps_MT)
        z_hat_MT = self.mask_latents(z_MT, eps_MT)
        # turning back into vector form
        z = z_MT.reshape(-1, self.latent_size*self.unit_dim)
        z_hat = z_hat_MT.reshape(-1, self.latent_size*self.unit_dim)
        return z, z_hat

    def forward_SCM_intervention(self, eps:Tensor, dim:int, value:Tensor, **kwargs):
        """Performs forward pass through SCM layer with intervention applied on dimension dim
        The intervention has to be applied twice: once on the inpput and once on the output
        nodes of the SCM layer
        Note: value has to be a Tensor of unit-dim size"""

        # turning into matrix form (nxk)
        eps_MT = eps.reshape(-1, self.latent_size, self.unit_dim)
        z_MT = self.DAG_transform(eps_MT)
        #apply intervention on the input
        z_MT[:,dim,:] = value.reshape(1,self.unit_dim)
        z_hat_MT = self.mask_latents(z_MT, eps_MT)
        #apply intervention on the output
        z_hat_MT[:,dim,:] = value.reshape(1,self.unit_dim)
        # turning back into vector form
        z = z_MT.reshape(-1, self.latent_size*self.unit_dim)
        z_hat = z_hat_MT.reshape(-1, self.latent_size*self.unit_dim)
        return z, z_hat

    def generate_intervened(self, inputs:Tensor, dim:int, value:Tensor, **kwargs):
        """Generates samples by applying the specified intervention on the inputs"""
        activate= kwargs.get('activate',False)
        eps, eps_mu, eps_v = self.encode(inputs, **kwargs)
        _, z_bar = self.forward_SCM_intervention(eps, dim, value, **kwargs)
        x_bar = self.decode(z_bar, activate=activate)
        return x_bar

    def forward(self, inputs: Tensor, **kwargs) -> list:
        activate = kwargs.get('activate',False)
        eps, eps_mu, eps_v = self.encode(inputs, **kwargs)
        z, z_hat = self.forward_SCM(eps)
        z_mu, z_mu_hat = self.forward_SCM(eps_mu)
        x_hat = self.decode(z_hat, activate=activate)
        return [x_hat, eps_mu, eps_v, z, z_hat, z_mu, z_mu_hat]

    def get_representation(self, inputs:Tensor, **kwargs):
        """ returns a representation vector for the given inputs"""
        eps, eps_mu, eps_v = self.encode(inputs, **kwargs)
        z, z_hat = self.forward_SCM(eps)
        return z

    def add_regularisation_terms(self, *args, **kwargs):
        """Implements the very cumbersome loss defined in the CausalVAE paper"""
        # x_hat = args[0]
        eps_mu = args[1]
        eps_v = args[2]
        z = args[3]
        z_hat = args[4]
        z_mu = args[5] #TODO: get rid of this one
        z_mu_hat = args[6]
        X = kwargs["X"]
        u = kwargs["latent_labels"]
        device = kwargs.get('device')
        losses = kwargs.get('losses')
        #Retrieving all the coefficients to mix the losses together
        alpha = kwargs["alpha"]
        beta = kwargs["beta"]
        gamma = kwargs["gamma"]
        KL_eps = KL_multiple_univariate_gaussians(eps_mu, torch.zeros_like(eps_mu).to(device),
                                                   eps_v, torch.zeros_like(eps_v).to(device),
                                                   reduce=True)
        losses['KL_eps'] = KL_eps
        KL_z = 0
        B = X.shape[0]
        for i in range(self.latent_size):
            low = i*self.unit_dim; high=(i+1)*self.unit_dim
            #TODO: check whether here we should be using another z
            # perhaps use z_hat or z_mu_hat
            KL_z += KL_multiple_univariate_gaussians(z_mu_hat[:,low:high], u[:,i].reshape(-1,1).to(device),
                                                     torch.zeros(B, self.unit_dim).to(device), torch.zeros(B, self.unit_dim).to(device),
                                                     reduce=True)
        losses['KL_z'] = KL_z/self.latent_size
        losses['KL_loss'] = KL_z + KL_eps

        #TODO: in the reference code they compute KL before and after the masking
        rec_z = F.mse_loss(z,z_hat)
        losses['rec_z'] = rec_z

        # labels reconstruction loss
        u_hat = torch.matmul(u, self.A) #note: u should have size NxK here (otherwise it's not matching A dimension)
        rec_u = F.mse_loss(u,u_hat)
        losses['rec_u'] = rec_u

        # DAG loss
        DAG_loss = continuous_DAG_constraint(self.A, self.latent_size)
        losses['DAG_loss'] = DAG_loss

        losses['loss'] = losses['loss'] + (KL_z + KL_eps) + alpha*DAG_loss + beta*rec_u + gamma*rec_z
        return losses

    def DAG_loss(self, u, gamma=0.25, eta=1.1):
        """
        Implements pre-training loss for causal graph layer
        The details are gathered from the appendix, section C.3
        The augmented Lagrangian objective makes use of several parameters (lamda, c) that
        are dynamically updated with the iterations. A memory has to be keps
        """
        lamda = self.DAG_loss_parameters["lamda"]
        c = self.DAG_loss_parameters["c"]
        u_hat = torch.matmul(u, self.A) #note: u should have size NxK here (otherwise it's not matching A dimension)
        lu = torch.mean(torch.norm((u-u_hat),p=2, dim=1))
        # DAG loss
        H = continuous_DAG_constraint(self.A, self.latent_size)
        loss = lu + lamda*H + (c/2)*torch.pow(H,2)
        # compute new values for lamda, c and threshold
        with torch.no_grad():
            #self.DAG_loss_parameters["lamda"] = lamda + c*H
            #if abs(H)>self.DAG_loss_parameters["threshold"]:
            #    self.DAG_loss_parameters["c"] = c*eta
            self.DAG_loss_parameters["threshold"] = gamma*abs(H)

        return loss, H, lu


    def loss_function(self, *args, **kwargs):
        """Overriding base loss_function to insert pre-training logic:"""

        losses = {}
        if kwargs.get("pretraining") >= kwargs.get("epoch",0): # pretraining
            u = kwargs["latent_labels"]
            loss, H, lu = self.DAG_loss(u)
            losses["H"] = H
            losses["labels MSE"] = lu
            losses["loss"] = loss
            return losses

        use_MSE = kwargs.get('use_MSE',True)
        X_hat = args[0]
        X = kwargs.get('X')
        MSE,BCE = pixel_losses(X,X_hat, act=self.act)
        L_rec = MSE if use_MSE else BCE

        losses['Reconstruction_loss'] = L_rec
        losses['loss'] = L_rec

        losses = self.add_regularisation_terms(*args, losses=losses, **kwargs)

        return losses






