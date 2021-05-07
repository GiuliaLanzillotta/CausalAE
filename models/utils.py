"""Implementation of various utilities """
import math

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

def Dittadi_weight_init_rule(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        torch.nn.init.xavier_uniform_(m.weight, gain=0.1)
        torch.nn.init.constant_(m.bias, -1.0)



