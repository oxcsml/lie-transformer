import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, reduce

from lie_conv.lieGroups import SE3
from lie_conv.lieConv import GlobalPool, Swish
from lie_conv.utils import Pass, Expression
from lie_conv.masked_batchnorm import MaskBatchNormNd
from eqv_transformer.multihead_neural import MultiheadWeightNet


class SumKernel(nn.Module):
    def __init__(self, location_kernel, feature_kernel):
        super().__init__()

        self.location_kernel = location_kernel
        self.feature_kernel = feature_kernel

    def forward(self, pairwise_locations, mask, query_features, key_features, nbhd_idx):
        return self.location_kernel((None, pairwise_locations, mask))[
            1
        ].squeeze() + self.feature_kernel(query_features, key_features, nbhd_idx)
        # return self.feature_kernel(query_features, key_features, nbhd_idx)


class DotProductKernel(nn.Module):
    def __init__(
        self, embed_dim, k_dim, q_dim, n_heads, k_bias=True, q_bias=True,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // n_heads
        assert (
            self.head_dim * self.n_heads == self.embed_dim
        ), "embed_dim must be divisible by n_heads"

        self.k_dim = k_dim
        self.q_dim = q_dim

        self.fc_k = nn.Linear(k_dim, embed_dim, bias=k_bias)
        self.fc_q = nn.Linear(q_dim, embed_dim, bias=q_bias)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_k.reset_parameters()
        self.fc_q.reset_parameters()

    def forward(self, k, q, nbhd_idx):
        """
        Parameters
        ----------
        query_f : torch.Tensor
            shape (bs, n, c_in)
        key_f : torch.Tensor
            shape (bs, n, m, c_in)

        Returns
        -------
        torch.Tensor
            shape (bs, n, m, h)
        """
        # (bs, m, c_in) -> (bs, m, embed_dim) -> (bs * n_heads, m, h_dim)
        K = rearrange(self.fc_k(k), "b n (h d) -> (b h) n d", h=self.n_heads)
        # (bs, n, c_in) -> (bs, n, embed_dim) -> (bs * n_heads, n, h_dim)
        Q = rearrange(self.fc_q(q), "b n (h d) -> (b h) n d", h=self.n_heads)
        # (bs * n_heads, n, h_dim), (bs * n_heads, m, h_dim) -> (bs * n_heads, n, m)
        A_ = Q.bmm(K.transpose(1, 2)) / math.sqrt(self.embed_dim)

        # (bs * n_heads, n, nbhd_size) -> (bs, n, nbhd_size, n_heads)
        A_ = rearrange(A_, "(b h) n m -> b n m h", h=self.n_heads)

        # Batch indicies
        B = (
            torch.arange(A_.shape[0], device=A_.device)
            .long()[:, None, None]
            .expand(*nbhd_idx.shape)
        )

        # Get NNS indexes
        NNS = (
            torch.arange(A_.shape[1], device=A_.device)
            .long()[None, :, None]
            .expand(*nbhd_idx.shape)
        )

        A_ = A_[B, NNS, nbhd_idx]

        return A_


class EquivairantMultiheadAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        n_heads,
        group,
        layer_norm=False,
        kernel_dim=16,
        act="swish",
        bn=False,
    ):

        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.n_heads = n_heads
        self.group = group

        self.kernel = SumKernel(
            MultiheadWeightNet(
                group.lie_dim + 2 * group.q_dim,
                1,
                n_heads,
                hid_dim=kernel_dim,
                act=act,
                bn=bn,
            ),
            DotProductKernel(c_in, c_in, c_in, n_heads=n_heads),
        )

        self.input_linear = nn.Linear(c_in, c_out)
        self.output_linear = nn.Linear(c_out, c_out)

        if layer_norm:
            self.ln_1 = nn.LayerNorm(c_out)
            self.ln_2 = nn.LayerNorm(c_out)

    def extract_neighbourhoods(self, input, query_indices=None):
        """ Extracts which points each other point is to attend to based on distance, or graph structure
        

        Parameters
        ----------
        input : (pairwise_g, coset_functions, mask)
        """
        # TODO: Currently no down sampling in this step.

        pairwise_g, coset_functions, mask = input

        coset_functions_at_query = coset_functions

        # TODO: temporarily selecting all points to be in neighbourhood
        nbhd_idx = (
            torch.arange(coset_functions.shape[1], device=coset_functions.device)
            .long()[None, None, :]
            .expand(pairwise_g.shape[:-1])
        )

        # Get batch index array
        BS = (
            torch.arange(coset_functions.shape[0], device=coset_functions.device)
            .long()[:, None, None]
            .expand(*nbhd_idx.shape)
        )
        # Get NNS indexes
        NNS = (
            torch.arange(coset_functions.shape[1], device=coset_functions.device)
            .long()[None, :, None]
            .expand(*nbhd_idx.shape)
        )

        nbhd_pairwise_g = pairwise_g[
            BS, NNS, nbhd_idx
        ]  # (bs, n * ns, n * ns, g_dim) -> (bs, n * ns, nbhd_size, g_dim)
        nbhd_coset_functions = coset_functions[
            BS, nbhd_idx
        ]  # (bs, n * ns, c_in) -> (bs, n * ns, nbhd_size, c_in)
        nbhd_mask = mask[BS, nbhd_idx]  # (bs, n * ns) -> (bs, n * ns, nbhd_size)

        # (bs, n * ns, nbhd_size, g_dim), (bs, n * ns, nbhd_size, c_in), (bs, n * ns, nbhd_size), (bs, n * ns, nbhd_size)
        return nbhd_pairwise_g, nbhd_coset_functions, nbhd_mask, nbhd_idx

    def forward(self, input):

        # (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_in), (bs, n * ns)
        pairwise_g, coset_functions, mask = input

        # (bs, n * ns, nbhd_size, g_dim), (bs, n * ns, nbhd_size, c_in), (bs, n * ns, nbhd_size), (bs, n * ns, nbhd_size)
        (
            nbhd_pairwise_g,
            nbhd_coset_functions,
            nbhd_mask,
            nbhd_idx,
        ) = self.extract_neighbourhoods(input)

        # Expand across head dimension TODO: possibly waseful and could avoid with a special linear layer
        # (bs, n * ns, nbhd_size, g_dim) -> (bs, n * ns, nbhd_size, h, g_dim)
        nbhd_pairwise_g = nbhd_pairwise_g.unsqueeze(-2).repeat(1, 1, 1, self.n_heads, 1)
        # Exapand the mask along the head dim
        nbhd_mask = nbhd_mask.unsqueeze(-1)

        # (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_in), (bs, n * ns, nbhd_size, c_in) -> (bs, n * ns, nbhd_size, h)
        presoftmax_weights = self.kernel(
            nbhd_pairwise_g, nbhd_mask, coset_functions, coset_functions, nbhd_idx
        )
        # presoftmax_weights = torch.exp(presoftmax_weights)
        # Zero out the correct channels
        presoftmax_weights = torch.where(
            # (bs, n * ns, nbhd_size) -> (bs, n * ns, nbhd_size, 1). Constant along head dim
            nbhd_mask,
            presoftmax_weights,
            torch.tensor(
                -1e38, dtype=presoftmax_weights.dtype, device=presoftmax_weights.device
            )
            * torch.ones_like(presoftmax_weights),
        )

        # Compute the normalised attention weights
        # (bs, n * ns, nbhd_size, h) -> (bs, n * ns, nbhd_size, h)
        # softmax_attention = presoftmax_weights / presoftmax_weights.sum(
        #     dim=(2), keepdim=True
        # )
        softmax_attention = F.softmax(presoftmax_weights, dim=2)

        # Pass the inputs through the value linear layer
        # (bs, n * ns, nbhd_size, c_in) -> (bs, n * ns, nbhd_size, c_out)
        nbhd_coset_functions = self.input_linear(nbhd_coset_functions)
        coset_functions = self.input_linear(coset_functions)
        # Split the features into heads
        nbhd_coset_functions = rearrange(
            nbhd_coset_functions, "b n m (h d) -> b n m h d", h=self.n_heads
        )
        coset_functions = rearrange(
            coset_functions, "b n (h d) -> b n h d", h=self.n_heads
        )

        # Sum over the coefficients
        # TODO: Currently allows self interaction in the attention sum. Some pre matrices?
        # (bs, n * ns, nbhd_size, h), (bs, n * ns, nbhd_size, h, c_out / h) -> (bs, n * ns, nbhd_size, h)
        coset_functions = coset_functions + (
            softmax_attention.unsqueeze(-1) * nbhd_coset_functions
        ).sum(dim=2)

        coset_functions = rearrange(coset_functions, "b n h d -> b n (h d)")

        # Apply the output matrix and possibly layernorm
        # (bs, n * ns, h * c_in) -> (bs, n * ns, h * c_in)
        coset_functions = (
            coset_functions
            if getattr(self, "ln_1", None) is None
            else self.ln_1(coset_functions)
        )
        # (bs, n * ns, h * c_in) -> (bs, n * ns, c_out)
        coset_functions = self.output_linear(coset_functions)
        # (bs, n * ns, c_out) -> (bs, n * ns, c_out)
        coset_functions = (
            coset_functions
            if getattr(self, "ln_2", None) is None
            else self.ln_2(coset_functions)
        )

        # ( (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_out), (bs, n * ns) )
        return (pairwise_g, coset_functions, mask)


class EquivariantTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden,
        num_layers,
        num_heads,
        layer_norm=False,
        global_pool=True,
        global_pool_mean=True,
        group=SE3(0.2),
        liftsamples=1,
        kernel_dim=16,
        kernel_act="swish",
        batch_norm=False,
    ):
        super().__init__()

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention = lambda c_in, c_out, n_head: EquivairantMultiheadAttention(
            c_in, c_out, n_head, group, layer_norm, kernel_dim, kernel_act, batch_norm
        )

        self.net = nn.Sequential(
            Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
            *[
                attention(dim_hidden[i], dim_hidden[i + 1], num_heads[i])
                for i in range(num_layers)
            ],
            # MaskBatchNormNd(dim_hidden[-1]) if batch_norm else nn.Sequential(),
            Pass(Swish() if kernel_act == "swish" else nn.ReLU()),
            Pass(nn.Linear(dim_hidden[-1], dim_output), dim=1),
            GlobalPool(mean=global_pool_mean)
            if global_pool
            else Expression(lambda x: x[1]),
        )

        self.group = group
        self.liftsamples = liftsamples

    def forward(self, input):
        lifted_data = self.group.lift(input, self.liftsamples)
        return self.net(lifted_data)
