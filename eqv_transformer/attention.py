import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


"""
Code in this file is a reimplementation of the Set Transformer from https://arxiv.org/abs/1810.00825
"""


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
            logits = presence_q * logits - (1.0 - presence_q) * inf

        if presence_k is not None:
            presence_k = presence_k.repeat(self.num_heads, 1).unsqueeze(-2)
            logits = presence_k * logits - (1.0 - presence_k) * inf

        A = torch.softmax(logits, 2)

        O = torch.cat((Q_ + A.bmm(V_)).split(queries.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SelfattentionBlock(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SelfattentionBlock, self).__init__()
        self.mab = MultiheadAttentionBlock(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, presence=None):
        return self.mab(X, X, presence_q=presence, presence_k=presence)


class InducedSelfAttentionBlock(nn.Module):
    """Like self-attention block, but uses inducing points to reduce computation complexity."""

    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(InducedSelfAttentionBlock, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MultiheadAttentionBlock(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MultiheadAttentionBlock(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, presence=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, presence_k=presence)
        return self.mab1(X, H, presence_q=presence)


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


class SetTransformer(nn.Module):
    """Builds a Set Transformer."""

    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        n_enc_layers=2,
        n_dec_layers=2,
        dim_hidden=128,
        num_heads=4,
        num_inducing_points=0,
        ln=False,
    ):
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

        Model introduced in Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks https://arxiv.org/abs/1810.00825
        """

        super(SetTransformer, self).__init__()

        if num_inducing_points > 0:
            enc_layer_class = functools.partial(
                InducedSelfAttentionBlock, num_inds=num_inducing_points
            )
        else:
            enc_layer_class = SelfattentionBlock

        enc_layers = [
            InputWrapper(enc_layer_class(dim_input, dim_hidden, num_heads, ln=ln))
        ]
        for _ in range(n_enc_layers - 1):
            enc_layers.append(
                InputWrapper(enc_layer_class(dim_hidden, dim_hidden, num_heads, ln=ln))
            )

        self.enc_layers = nn.ModuleList(enc_layers)
        # self.enc = enc_layers[0]

        self.pooling_layer = InputWrapper(
            MultiheadAttentionPooling(dim_hidden, num_heads, num_outputs, ln=ln)
        )

        dec_layers = []
        for _ in range(n_dec_layers):
            dec_layers.append(
                InputWrapper(
                    SelfattentionBlock(dim_hidden, dim_hidden, num_heads, ln=ln)
                )
            )
        dec_layers.append(InputWrapper(nn.Linear(dim_hidden, dim_output)))

        self.dec_layers = nn.ModuleList(dec_layers)
        # self.dec = dec_layers[0]

    def forward(self, X, presence=None):
        for enc_layer in self.enc_layers:
            X = enc_layer([X, presence])

        X = self.pooling_layer([X, presence])

        for dec_layer in self.dec_layers:
            X = dec_layer(X)

        return X