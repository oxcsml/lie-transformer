import torch
from torch import nn


class Normalize(nn.Module):
    """Various normalization schemes."""

    def __init__(self, norm_type='none', n_hidden=0):
        super(Normalize, self).__init__()
        self.norm_type = norm_type

        if self.norm_type == 'layer_norm':
            assert n_hidden > 0, '`n_hidden` cannot be zero for `layer_norm`.'
            self.ln = nn.LayerNorm(n_hidden)

    def forward(self, x):
        if self.norm_type == 'layer_norm':
            return self.ln(x)
        elif self.norm_type == 'norm':
            return x / torch.norm(x, dim=-1, keepdim=True)
        elif self.norm_type == 'none':
            return x
        else:
            raise NotImplementedError('Invalid norm type "{}".'.format(self.norm_type))