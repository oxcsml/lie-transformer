import math

import torch
from torch import nn

from lie_conv.lieConv import Swish
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Pass


class MultiheadLinear(nn.Module):
    """ Layer to perform n_heads MLPs in parallel for multihead kernels

        Parameters
        ----------
        n_heads : int
            number of heads
        c_in : int
            input dimension
        c_out : int
            output dimension
        bias : bool, optional
            use a bias parameter, by default True
    """

    def __init__(self, n_heads, c_in, c_out, bias=True):
        super(MultiheadLinear, self).__init__()

        self.n_heads = n_heads
        self.c_in = c_in
        self.c_out = c_out

        # n_head linear transforms
        self.weight = nn.Parameter(torch.Tensor(self.n_heads, self.c_in, self.c_out))

        if bias:
            # n_head biases
            self.bias = nn.Parameter(torch.Tensor(self.n_heads, self.c_out))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Uses einsum notation to get the correct output
        if self.bias is not None:
            return torch.einsum("...hi,hio->...ho", input, self.weight) + self.bias
        else:
            return torch.einsum("...hi,hio->...ho", input, self.weight)


def MultiheadLinearBNact(c_in, c_out, n_heads, act="swish", bn=True):
    """??(from LieConv - not sure it does assume) assumes that the inputs to the net are shape (bs,n,mc_samples,c)"""
    assert act in ("relu", "swish"), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(c_out)
    return nn.Sequential(
        Pass(MultiheadLinear(n_heads, c_in, c_out), dim=1),
        normlayer if bn else nn.Sequential(),
        Pass(Swish() if act == "swish" else nn.ReLU(), dim=1),
    )


def MultiheadWeightNet(in_dim, out_dim, n_heads, act, bn, hid_dim=32):
    return nn.Sequential(
        *MultiheadLinearBNact(in_dim, hid_dim, n_heads, act, bn),
        *MultiheadLinearBNact(hid_dim, hid_dim, n_heads, act, bn),
        *MultiheadLinearBNact(hid_dim, out_dim, n_heads, act, bn),
    )

