"""Implementation of various utilities """
from torch import nn
import torch
import torch.nn.functional as F


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
    if name=="LR":return nn.LeakyReLU
    raise  NotImplementedError(f"Specified activation -{name}- not supported")



