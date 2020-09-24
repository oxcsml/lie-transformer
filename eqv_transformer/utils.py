import torch
import torch.nn as nn

from lie_conv.utils import Expression


class ResidualBlock(nn.Module):
    def __init__(self, module, dim=None):
        super().__init__()
        self.module = module
        self.dim = dim

    def forward(self, input):
        if self.dim is None:
            return input + self.module(input)
        else:
            input[self.dim] = self.module(input)[self.dim]
            return input


class GroupLift(nn.Module):
    def __init__(self, group, liftsamples=1):
        super().__init__()
        self.group = group
        self.liftsamples = liftsamples

    def forward(self, x):
        return self.group.lift(x, self.liftsamples)


class GlobalPool(nn.Module):
    """computes values reduced over all spatial locations (& group elements) in the mask"""

    def __init__(self, mean=False):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        """x [xyz (bs,n,d), vals (bs,n,c), mask (bs,n)]"""
        if len(x) == 2:
            return x[1].mean(1)
        coords, vals, mask = x
        summed = torch.where(mask.unsqueeze(-1), vals, torch.zeros_like(vals)).sum(1)
        if self.mean:
            summed_mask = mask.sum(-1).unsqueeze(-1).clamp(min=1)
            summed /= summed_mask

        return summed


def Swish():
    return Expression(lambda x: x * torch.sigmoid(x))
