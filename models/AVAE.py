"""Implementation of AVAE model as in https://arxiv.org/abs/2012.03715"""
import torch
from torch import Tensor, nn

from . import VAE

class AVAE(VAE):
    """VAE + autoencoding self-consistency regularisation"""

    def __init__(self, params):
        super().__init__(params)
        self.ro = params.get('ro')
        self.LogGaussianError = nn.GaussianNLLLoss(reduction='none')

    def forward(self, inputs: Tensor, **kwargs) -> list:
        activate= kwargs.get('activate',False)
        z, mu, logvar = self.encode(inputs)
        x_hat = self.decode(z, activate)
        return  [x_hat, mu, logvar, z]

    def compute_autoencoding_loss(self, z, device):
        """Component of the loss term due to autoencoding consistency
        The expectations are approximated using Monte Carlo estimates with number of samples = 1"""
        with torch.no_grad():
            x_tilde = self.decode(z, activate=True)
        z_tilde, mu_tilde, logvar_tilde = self.encode(x_tilde)
        #all vectors have dimension (N,D)
        variances = torch.ones_like(z_tilde, device=device)*(1-self.ro**2)
        d_z_z_tilde = self.LogGaussianError(z_tilde, z*self.ro, variances)
        p_z_tilde = self.LogGaussianError(z_tilde, mu_tilde, logvar_tilde.exp())
        loss =  (d_z_z_tilde-p_z_tilde).sum(dim=1).mean()
        return loss

    def add_regularisation_terms(self, *args, **kwargs):
        """ Takes as input the losses dictionary containing the reconstruction
        loss and adds all the regularisation terms to it"""
        z = args[3]
        device = kwargs.get('device')
        KL_weight = kwargs["KL_weight"]
        losses = VAE.add_regularisation_terms(self, *args, **kwargs)
        ACN_loss = self.compute_autoencoding_loss(z, device)
        losses['ACN'] = ACN_loss
        losses['loss'] += self.beta * KL_weight * ACN_loss
        return losses
