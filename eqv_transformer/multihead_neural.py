import math
from collections import OrderedDict

import torch
from torch import nn

from lie_conv.lieConv import Swish
from lie_conv.masked_batchnorm import MaskBatchNormNd
from lie_conv.utils import Pass


activation_fn = {
    "swish": Swish,
    "relu": nn.ReLU,
    "softplus": nn.Softplus,
}


class MultiheadLinear(nn.Module):
    """Layer to perform n_heads MLPs in parallel for multihead kernels

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
        for i in range(self.n_heads):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias[i], -bound, bound)

    def forward(self, input):
        # Uses einsum notation to get the correct output
        if self.bias is not None:
            return torch.einsum("...hi,hio->...ho", input, self.weight) + self.bias
        else:
            return torch.einsum("...hi,hio->...ho", input, self.weight)


def MultiheadLinearBNact(c_in, c_out, n_heads, act="swish", bn=True):
    """??(from LieConv - not sure it does assume) assumes that the inputs to the net are shape (bs,n,mc_samples,c)"""
    assert act in ("relu", "swish", "softplus"), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(c_out)
    return nn.Sequential(
        OrderedDict(
            [
                ("linear", Pass(MultiheadLinear(n_heads, c_in, c_out), dim=1)),
                ("norm", normlayer if bn else nn.Sequential()),
                ("activation", Pass(activation_fn[act](), dim=1)),
            ]
        )
    )


def MultiheadWeightNet(in_dim, out_dim, n_heads, act, bn, hid_dim=32):
    # TODO: Check speed difference
    # return nn.Sequential(
    #     *MultiheadLinearBNact(in_dim, hid_dim, n_heads, act, bn),
    #     *MultiheadLinearBNact(hid_dim, hid_dim, n_heads, act, bn),
    #     *MultiheadLinearBNact(hid_dim, out_dim, n_heads, act, bn),
    # )
    return nn.Sequential(
        OrderedDict(
            [
                (
                    "LinNormAct_1",
                    MultiheadLinearBNact(in_dim, hid_dim, n_heads, act, bn),
                ),
                (
                    "LinNormAct_2",
                    MultiheadLinearBNact(hid_dim, hid_dim, n_heads, act, bn),
                ),
                (
                    "LinNormAct_3",
                    MultiheadLinearBNact(hid_dim, out_dim, n_heads, act, bn),
                ),
            ]
        )
    )


def MultiheadMLP(in_dim, hid_dim, out_dim, n_heads, n_layers, act, bn):
    if n_layers == 1:
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        "LinNormAct_1",
                        MultiheadLinearBNact(in_dim, out_dim, n_heads, act, bn),
                    )
                ]
            )
        )
    else:
        layers = []
        layers.append(
            ("LinNormAct_1", MultiheadLinearBNact(in_dim, hid_dim, n_heads, act, bn))
        )
        for i in range(1, n_layers):
            layers.append(
                (
                    f"LinNormAct_{i+1}",
                    MultiheadLinearBNact(hid_dim, hid_dim, n_heads, act, bn),
                )
            )
        layers.append(
            (
                f"LinNormAct_{n_layers}",
                MultiheadLinearBNact(hid_dim, out_dim, n_heads, act, bn),
            )
        )
        return nn.Sequential(OrderedDict(layers))


def LinearBNact(chin, chout, act="swish", bn=True):
    """assumes that the inputs to the net are shape (bs,n,mc_samples,c)"""
    assert act in ("relu", "swish", "softplus"), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(chout)
    return nn.Sequential(
        OrderedDict(
            [
                ("linear", Pass(nn.Linear(chin, chout), dim=1)),
                ("norm", normlayer if bn else nn.Sequential()),
                ("activation", Pass(activation_fn[act](), dim=1)),
            ]
        )
    )


def MLP(dim_in, dim_hid, dim_out, num_layers, act, bn):
    if num_layers == 1:
        return nn.Sequential(
            OrderedDict([("LinNormAct_1", LinearBNact(dim_in, dim_out, act, bn))])
        )
    else:
        layers = []
        layers.append(("LinNormAct_1", LinearBNact(dim_in, dim_hid, act, bn)))
        for i in range(1, num_layers):
            layers.append((f"LinNormAct_{i+1}", LinearBNact(dim_hid, dim_hid, act, bn)))
        layers.append(
            (f"LinNormAct_{num_layers}", LinearBNact(dim_hid, dim_out, act, bn))
        )
        return nn.Sequential(OrderedDict(layers))
