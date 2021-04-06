# Code for beta-VAE
# Paper: https://openreview.net/forum?id=Sy2fzU9gl 
import numpy as np
from torch import nn

class VAE(nn.Module):

    def __init__(self, dim_in, latent_dim, channels_list:List=None) -> None:
        super(VAE, self).__init__()
        C, H, W = dim_in
        self.latent_dim = latent_dim
        if channels_list is None:
            channels_list = [32, 64, 128, 256, 512]
        # Building encoder 
        modules = []
        for c in channels_list:
            modules.append(
                # Conv block - 
                #TODO: create a separate layer for this
                nn.Sequential(
                    nn.Conv2d(C, out_channels=c,kernel_size= 2, stride= 2, padding  = 1),
                    nn.BatchNorm2d(c),
                    nn.LeakyReLU())
            )
            final_channels = c
        self.encoder = nn.Sequential(*modules)
        # 
        self.fc_mu = nn.Linear(channels_list[-1]*4, latent_dim)
        self.fc_var = nn.Linear(channels_list[-1]*4, latent_dim)

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size:int, current_device: int, **kwargs) -> Tensor:
        raise RuntimeWarning()

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass