# Code for Ladder VAE 
# Paper: https://arxiv.org/pdf/1602.02282.pdf
import numpy as np
from torch import nn

class LadderVAE(nn.Module):

    def __init__(self):
        # Initialise class parameters
        # Initialise encoder and decoder 
        self.encoder_layers = nn.ModuleList([])
        self.generator_layers = nn.ModuleList([])
        # Define which likelihood to use

    def forward(self, x):
        # input preprocessing (e.g. padding)
        # bottom-up pass: from observable to latent space
        code = self.encode(x)
        # top-down generation: from latent to observable
        x_hat = self.generate(code)
        # post-processing (e.g. cropping)
        # compute loss: log-likelihood and kl 
        # return output, loss and other info

    def encode(self, x):
        pass

    def generate(self, code):
        pass

    def sample_prior(self):
        pass
