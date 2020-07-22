import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from math import *


class MultiheadAttentionBlock(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MultiheadAttentionBlock, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, queries, keys, presence_q=None, presence_k=None):
        queries = self.fc_q(queries)
        keys, values = self.fc_k(keys), self.fc_v(keys)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(queries.split(dim_split, 2), 0)
        K_ = torch.cat(keys.split(dim_split, 2), 0)
        V_ = torch.cat(values.split(dim_split, 2), 0)

        logits = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)

        # bias the logits to not take absent entries into account
        inf = torch.tensor(1e38, dtype=torch.float32, device=queries.device)
        if presence_q is not None:
            presence_q = presence_q.repeat(self.num_heads, 1).unsqueeze(-1)
            logits = presence_q * logits - (1. - presence_q) * inf

        if presence_k is not None:
            presence_k = presence_k.repeat(self.num_heads, 1).unsqueeze(-2)
            logits = presence_k * logits - (1. - presence_k) * inf

        A = torch.softmax(logits, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(queries.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class SelfattentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SelfattentionBlock, self).__init__()
        self.mab = MultiheadAttentionBlock(
            dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, presence=None):
        return self.mab(X, X, presence_q=presence, presence_k=presence)


class MultiheadAttentionPooling(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(MultiheadAttentionPooling, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MultiheadAttentionBlock(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, presence=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, presence_k=presence)


class InputWrapper(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, args):

        if isinstance(args, (list, tuple)):
            return self.module(*args)
        return self.module(args)


class Embed(nn.Module):
    """Lifts data-content pairs to partial function over group

    Args:
        cn: size of cyclic group (representing cn rotations)
    """

    def __init__(self, dim_input, cn, dim_hidden, content_type):
        super(Embed, self).__init__()

        self.content_type = content_type
        self.cn = cn
        self.mlp = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden)
        )

        if content_type not in ['centroidal', 'constant', 'pairwise_distances']:
            raise NotImplementedError(
                f'Content type {content_type} has not been implemented yet.')

        if content_type == 'centroidal':
            dim_input = 2
        elif content_type == 'constant':
            dim_input = 1
        elif content_type == 'pairwise_distances':
            pass  # i.e. use the arg dim_input

    def matrixify(self, X):
        angles = 2*pi * X[..., [2]] / self.cn
        cosines = torch.cos(angles)
        sines = torch.sin(angles)

        rotations_1 = torch.cat([cosines, -sines], dim=2).unsqueeze(2)
        rotations_2 = torch.cat([sines, cosines], dim=2).unsqueeze(2)
        rotations = torch.cat([rotations_1, rotations_2], dim=2)

        X_lift = torch.cat(
            [rotations, X[..., :2].unsqueeze(2).transpose(2, 3)], dim=3)
        X_lift = torch.cat(
            [X_lift, torch.ones_like(X_lift)[:, :, :1, :]], dim=2)
        X_lift[:, :, [2], :2] = 0.

        return X_lift, rotations

    def forward(self, X, mask):
        """
        Args:
            X: shape = (bs, num_pairs, input_location_dim)
            Y: shape = (bs, num_pairs, input_content_dim)
        """

        X_lift = torch.cat([torch.cat(
            [X, i*torch.ones_like(X)[..., :1]], dim=2) for i in range(self.cn)], dim=1)

        mask_lift = torch.cat([mask for _ in range(self.cn)], dim=1)

        if self.content_type == 'centroidal':

            centroids = X.mean(dim=1, keepdim=True)
            X_c = X - centroids

            _, rotations = self.matrixify(-X_lift)
            X_c_rep = X_c.repeat(1, self.cn, 1)

            Y_lift = rotations.view(-1, 2, 2).bmm(X_c_rep.unsqueeze(3).view(-1, 2, 1)).view(
                X_c_rep.shape[0], X_c_rep.shape[1], 2, 1).squeeze(3)
            Y_lift = self.mlp(Y_lift)

        if self.content_type == 'constant':

            Y = torch.ones_like(X)[..., :1]
            Y = self.mlp(Y)

            Y_lift = torch.cat([Y for _ in range(self.cn)], dim=1)

        if self.content_type == 'pairwise_distances':

            X_pairs = X.unsqueeze(2).transpose(1, 2) - X.unsqueeze(2)
            X_distances = torch.norm(X_pairs, dim=-1)
            X_distances = torch.sort(
                X_distances, dim=-1, descending=True)[0][..., :-1]

            Y = self.mlp(X_distances)

            Y_lift = torch.cat([Y for _ in range(self.cn)], dim=1)

        return X_lift, Y_lift, mask_lift


class EqvSelfAttention(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(EqvSelfAttention, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)
        self.k_x = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )

    def forward(self, X_lift, Y_lift, X_pairs, presence_q=None, presence_k=None):
        keys = queries = values = Y_lift

        queries = self.fc_q(queries)
        keys, values = self.fc_k(keys), self.fc_v(keys)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(queries.split(dim_split, 2), 0)
        K_ = torch.cat(keys.split(dim_split, 2), 0)
        V_ = torch.cat(values.split(dim_split, 2), 0)

        logits = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        loc_logits = self.k_x(X_pairs).squeeze(3)

        # print(logits.shape, loc_logits.shape)

        logits = logits + loc_logits.repeat(self.num_heads, 1, 1)

        # bias the logits to not take absent entries into account
        inf = torch.tensor(1e38, dtype=torch.float32, device=queries.device)
        if presence_q is not None:
            presence_q = presence_q.repeat(self.num_heads, 1).unsqueeze(-1)
            logits = presence_q * logits - (1. - presence_q) * inf

        if presence_k is not None:
            presence_k = presence_k.repeat(self.num_heads, 1).unsqueeze(-2)
            logits = presence_k * logits - (1. - presence_k) * inf

        A = torch.softmax(logits, 2)

        # TODO: I think this Q_ below should be V_? 
        O = torch.cat((Q_ + A.bmm(V_)).split(queries.size(0), 0), 2)
        O = O if getattr(self, 'ln0', None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, 'ln1', None) is None else self.ln1(O)
        return O


class EqvSelfAttentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(EqvSelfAttentionBlock, self).__init__()
        self.mab = EqvSelfAttention(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X_lift, Y_lift, X_pairs, presence=None):
        return self.mab(X_lift, Y_lift, X_pairs, presence_q=presence, presence_k=presence)


class EqvTransformer(nn.Module):
    """Builds an Equivariant Transformer."""

    def __init__(self, dim_input, num_outputs, dim_output, content_type,
                 n_enc_layers=4, n_dec_layers=4,
                 dim_hidden=128, num_heads=4, ln=False, cn=5):
        """Build the module.

        Args:
            dim_input: int, input dimensionality.
            num_outputs: int, number of output vectors.
            dim_output: int, dimensionality of each output vector.
            n_enc_layers: int, number of layers in the encoder.
            n_dec_Layers: int, number of layers in the decoder.
            dim_hidden: int, dimensionality of each hidden vector.
            num_heads: int, number of attention heads.
            num_inducing_points: int, uses inducing points if > 0.
            ln: bool, uses Layer Normalization if True.
        """

        super(EqvTransformer, self).__init__()

        self.content_type = content_type
        self.cn = cn
        self.embedder = Embed(dim_input=dim_input, cn=cn,
                              dim_hidden=dim_hidden, content_type=content_type)

        enc_layers = []
        for _ in range(n_enc_layers):
            enc_layers.append(EqvSelfAttentionBlock(
                dim_hidden, dim_hidden, num_heads, ln=ln))

        self.enc_layers = nn.ModuleList(enc_layers)
        # self.enc = enc_layers[0]

        self.pooling_layer = InputWrapper(MultiheadAttentionPooling(
            dim_hidden, num_heads, num_outputs, ln=ln))

        dec_layers = []
        for _ in range(n_dec_layers):
            dec_layers.append(InputWrapper(SelfattentionBlock(
                dim_hidden, dim_hidden, num_heads, ln=ln)))
        dec_layers.append(InputWrapper(nn.Linear(dim_hidden, dim_output)))

        self.dec_layers = nn.ModuleList(dec_layers)
        # self.dec = dec_layers[0]

    def matrixify(self, X):
        angles = 2*pi * X[..., [2]] / self.cn
        cosines = torch.cos(angles)
        sines = torch.sin(angles)

        rotations_1 = torch.cat([cosines, -sines], dim=2).unsqueeze(2)
        rotations_2 = torch.cat([sines, cosines], dim=2).unsqueeze(2)
        rotations = torch.cat([rotations_1, rotations_2], dim=2)

        X_lift = torch.cat(
            [rotations, X[..., :2].unsqueeze(2).transpose(2, 3)], dim=3)
        X_lift = torch.cat(
            [X_lift, torch.ones_like(X_lift)[:, :, :1, :]], dim=2)
        X_lift[:, :, [2], :2] = 0.

        return X_lift, rotations

    def group_pairs(self, X_lift):
        # _, rotations = self.matrixify(X_lift)

        X_pairs = X_lift.unsqueeze(2).transpose(1, 2) - X_lift.unsqueeze(2)

        _, rotations_inverse = self.matrixify(-X_lift)
        rotations_inverse_repeated = rotations_inverse.unsqueeze(
            2).repeat(1, 1, rotations_inverse.shape[1], 1, 1)

        X_pairs[..., :2] = rotations_inverse_repeated.view(-1, 2, 2).bmm(X_pairs[..., :2].unsqueeze(
            4).view(-1, 2, 1)).view(*X_pairs[..., :2].unsqueeze(4).shape).squeeze(4)
        X_pairs[..., [-1]] = torch.remainder(X_pairs[..., [-1]], self.cn)
        return X_pairs

    def forward(self, X, presence=None):
        X_lift, Y_lift, presence_lift = self.embedder(X, presence)
        X_pairs = self.group_pairs(X_lift)

        for enc_layer in self.enc_layers:
            Y_lift = enc_layer(X_lift, Y_lift, X_pairs, presence_lift)

        Y_invariant = self.pooling_layer([Y_lift, presence_lift])

        for dec_layer in self.dec_layers:
            Y_invariant = dec_layer(Y_invariant)

        return Y_invariant
