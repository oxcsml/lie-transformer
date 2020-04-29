import numpy as np
import torch
import torch.nn as nn
import sys
import random

from eqv_transformer.masked_batchnorm import MaskBatchNormNd


def export(fn):
    """ Adds a function to the magic variable __all__ 
    used for import *.
    
    Parameters
    ----------
    fn : function
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, "__all__"):
        mod.__all__.append(fn.__name__)
    else:
        mod.__all__ = [fn.__name__]
    return fn


export(export)


@export
class Expression(nn.Module):
    """ Wrap a function in a torch module
    
    Parameters
    ----------
    func : single argument python function

    Repurposed from https://github.com/mfinzi/LieConv
    """

    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


@export
class Named(type):
    """ Pleasant metaclass for nicely naming classes
    
    Repurposed from https://github.com/mfinzi/LieConv
    """

    def __str__(self):
        return self.__name__

    def __repr__(self):
        return self.__name__


@export
class Pass(nn.Module):
    """ Helper class that takes a tuple and applies a module to a specific entry.
    Example usage for this project is taking (embedding, coset_functions, mask) and 
    applying a function just to the coset_functions. 

    Repurposed from https://github.com/mfinzi/LieConv
    """

    def __init__(self, module, dim=1):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, x):
        xs = list(x)
        xs[self.dim] = self.module(xs[self.dim])
        return tuple(xs)


@export
def Swish():
    """ Return the Swish activation function as a module. https://arxiv.org/pdf/1710.05941v1.pdf

    Repurposed from https://github.com/mfinzi/LieConv
    """
    return Expression(lambda x: x * torch.sigmoid(x))


@export
def LinearBNact(chin, chout, act="swish", bn=True):
    """Returns a pytorch module containing a linear layer with a nonlinearity and
    possibly batch normalisations

    Parameters
    ----------
    chin : Int
        Number of input channels.
    chout : Int
        Number of output channels.
    act : str, optional
        Which type of activation to use, by default "swish"
    bn : bool, optional
        Whether to use batch normalisation, by default True
    """
    assert act in ("ReLU", "Swish"), f"unknown activation type {act}"
    normlayer = MaskBatchNormNd(chout)
    return nn.Sequential(
        nn.Linear(chin, chout),
        normlayer if bn else nn.Sequential(),
        Swish() if act == "Swish" else nn.ReLU(),
    )
