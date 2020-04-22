import numpy as np
import torch
import torch.nn as nn
import sys
import random

def export(fn):
    """ Adds a function to the magic variable __all__ 
    used for import *.
    
    Parameters
    ----------
    fn : function
    """
    mod = sys.modules[fn.__module__]
    if hasattr(mod, '__all__'):
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

