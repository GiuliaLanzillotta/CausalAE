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




