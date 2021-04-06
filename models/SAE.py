# Code for SAE (no attention in the encoder)
# Paper: https://arxiv.org/abs/2006.07796
import numpy as np
from torch import nn

class SAE(nn.Module):
    
    def __init__(self):
        # Initialise class parameters
        # Initialise encoder and decoder 
        self.encoder_layers = nn.ModuleList([])
        # https://github.com/felixludos/learn_rep/blob/master/src/baseline.py 
        self.SCM_layers = nn.ModuleList([])
        self.generator_layers = nn.ModuleList([])
        # Define which likelihood to use

    def forward(self, x):
        # input preprocessing (e.g. padding)
        # bottom-up pass: from observable to latent space
        code = self.encode(x)
        # SCM pass: obtain causal variables from latent noise
        causal_code = self.SCM(code)
        # top-down generation: from causal variables to observable
        x_hat = self.generate(causal_code)
        # post-processing (e.g. cropping)
        # compute loss: log-likelihood and kl 
        # return output, loss and other info

    def encode(self, x):
        pass
    
    def SCM(self, code):
        pass

    def generate(self, causal_code):
        pass

    def sample_prior(self):
        pass
