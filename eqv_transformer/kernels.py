import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops.einops import rearrange, reduce

from lie_conv.lieGroups import SE3
from lie_conv.lieConv import Swish
from lie_conv.utils import Pass, Expression
from lie_conv.masked_batchnorm import MaskBatchNormNd
from eqv_transformer.multihead_neural import (
    MultiheadWeightNet,
    MultiheadMLP,
    LinearBNact,
    MLP,
)


class AttentionKernel(nn.Module):
    def __init__(
        self,
        feature_dim,
        location_dim,
        n_heads,
        feature_featurisation="dot_product",
        location_featurisation="mlp",
        location_feature_combination="sum",
        normalisation="none",
        hidden_dim=16,
        feature_embed_dim=None,
        activation="swish",
    ):

        super().__init__()

        if feature_embed_dim is None:
            feature_embed_dim = int(feature_dim / (4 * n_heads))
            print(feature_embed_dim)

        if feature_featurisation == "dot_product":
            self.feature_featurisation = DotProductKernel(
                feature_dim, feature_dim, feature_dim, n_heads
            )
            featurised_feature_dim = 1
        elif feature_featurisation == "linear_concat":
            self.feature_featurisation = LinearConcatEmbedding(
                int(feature_embed_dim * n_heads / 2), feature_dim, feature_dim, n_heads
            )
            featurised_feature_dim = feature_embed_dim
        elif feature_featurisation == "linear_concat_linear":
            featurised_feature_dim = feature_embed_dim
            self.feature_featurisation = LinearConcatLinearEmbedding(
                feature_embed_dim * n_heads, feature_dim, feature_dim, n_heads
            )
        else:
            raise ValueError(
                f"{feature_featurisation} is not a valid feature featurisation"
            )

        if location_featurisation == "mlp":
            self.location_featurisation = nn.Sequential(
                Expression(
                    lambda x: (
                        x[0],
                        x[1].unsqueeze(-2).repeat(1, 1, 1, n_heads, 1),
                        x[2],
                    )
                ),
                MultiheadWeightNet(
                    location_dim,
                    1,
                    n_heads,
                    hid_dim=hidden_dim,
                    act=activation,
                    bn=False,
                ),
                Expression(lambda x: x[1].squeeze(-1)),
            )
            featurised_location_dim = 1
        elif location_featurisation == "none":
            self.location_featurisation = Expression(
                lambda x: x[1].unsqueeze(-2).repeat(1, 1, 1, n_heads, 1)
            )
            featurised_location_dim = location_dim
        else:
            raise ValueError(
                f"{location_featurisation} is not a valid location featurisation"
            )

        if location_feature_combination == "sum":
            self.location_feature_combination = Expression(lambda x: x[0] + x[1])
        elif location_feature_combination == "mlp":
            self.location_feature_combination = nn.Sequential(
                Expression(lambda x: (None, torch.cat(x, dim=-1), None)),
                MultiheadMLP(
                    featurised_feature_dim + featurised_location_dim,
                    hidden_dim,
                    1,
                    n_heads,
                    3,
                    activation,
                    False,
                ),
                Expression(lambda x: x[1].squeeze(-1)),
            )
        elif location_feature_combination == "multiply":
            self.location_feature_combination = Expression(lambda x: x[0] * x[1])
        else:
            raise ValueError(
                f"{location_feature_combination} is not a valid combination method"
            )

        if normalisation == "none":
            self.normalisation = lambda attention_coeffs, mask: attention_coeffs
        elif normalisation == "softmax":

            def attention_func(attention_coeffs, mask):
                attention_coeffs = torch.where(
                    mask.unsqueeze(-1),
                    attention_coeffs,
                    torch.tensor(
                        -1e38,
                        dtype=attention_coeffs.dtype,
                        device=attention_coeffs.device,
                    )
                    * torch.ones_like(attention_coeffs),
                )
                return F.softmax(attention_coeffs, dim=2)

            self.normalisation = attention_func
        elif normalisation == "dot_product":

            def attention_func(attention_coeffs, mask):
                attention_coeffs = torch.where(
                    mask.unsqueeze(-1),
                    attention_coeffs,
                    torch.tensor(
                        0.0,
                        dtype=attention_coeffs.dtype,
                        device=attention_coeffs.device,
                    )
                    * torch.ones_like(attention_coeffs),
                )

                normalization = mask.unsqueeze(-1).sum(-2, keepdim=True)
                normalization = torch.clamp(normalization, min=1)
                return attention_coeffs / normalization

            self.normalisation = attention_func

    def forward(
        self, nbhd_pairwise_g, nbhd_mask, key_features, query_features, nbhd_idx
    ):
        feature_features = self.feature_featurisation(
            key_features, query_features, nbhd_idx
        )
        location_features = self.location_featurisation(
            (None, nbhd_pairwise_g, nbhd_mask)
        )

        if len(feature_features.shape) == 4:
            feature_features = feature_features.unsqueeze(-1)
        if len(location_features.shape) == 4:
            location_features = location_features.unsqueeze(-1)

        attention_weights = self.location_feature_combination(
            [feature_features, location_features]
        )
        return self.normalisation(attention_weights, nbhd_mask)


class SumKernel(nn.Module):
    def __init__(self, location_kernel, feature_kernel, n_heads):
        super().__init__()

        self.location_kernel = location_kernel
        self.feature_kernel = feature_kernel
        self.n_heads = n_heads

    def forward(self, pairwise_locations, mask, query_features, key_features, nbhd_idx):
        # Expand across head dimension TODO: possibly wasteful and could avoid with a special linear layer
        # (bs, n * ns, nbhd_size, g_dim) -> (bs, n * ns, nbhd_size, h, g_dim)
        pairwise_locations = pairwise_locations.unsqueeze(-2).repeat(
            1, 1, 1, self.n_heads, 1
        )
        # Exapand the mask along the head dim
        mask = mask.unsqueeze(-1)

        return self.location_kernel((None, pairwise_locations, mask))[1].squeeze(
            -1
        ) + self.feature_kernel(key_features, query_features, nbhd_idx)
        # return self.feature_kernel(query_features, key_features, nbhd_idx)


class LinearConcatEmbedding(nn.Module):
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
        # (bs, m, c_in) -> (bs, m, embed_dim) -> (bs * n_heads, m, h_dim)
        K = rearrange(self.fc_k(k), "b n (h d) -> b n h d", h=self.n_heads)
        # (bs, n, c_in) -> (bs, n, embed_dim) -> (bs * n_heads, n, h_dim)
        Q = rearrange(self.fc_q(q), "b n (h d) -> b n h d", h=self.n_heads)
        # Key features are just the same for each point
        K = K.unsqueeze(2).repeat(1, 1, nbhd_idx.shape[2], 1, 1)
        # Batch indices
        B = (
            torch.arange(Q.shape[0], device=Q.device)
            .long()[:, None, None]
            .expand(*nbhd_idx.shape)
        )
        # Extract the points for each nbhd
        Q = Q[B, nbhd_idx]

        # Concat and return
        return torch.cat([K, Q], dim=-1)


class LinearConcatLinearEmbedding(nn.Module):
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
        self.fc_o = nn.Linear(2 * self.head_dim, self.head_dim)

        self.reset_parameters()

    def reset_parameters(self):
        self.fc_k.reset_parameters()
        self.fc_q.reset_parameters()

    def forward(self, k, q, nbhd_idx):
        # (bs, m, c_in) -> (bs, m, embed_dim) -> (bs * n_heads, m, h_dim)
        K = rearrange(self.fc_k(k), "b n (h d) -> b n h d", h=self.n_heads)
        # (bs, n, c_in) -> (bs, n, embed_dim) -> (bs * n_heads, n, h_dim)
        Q = rearrange(self.fc_q(q), "b n (h d) -> b n h d", h=self.n_heads)
        # Key features are just the same for each point
        K = K.unsqueeze(2).repeat(1, 1, nbhd_idx.shape[2], 1, 1)
        # Batch indices
        B = (
            torch.arange(Q.shape[0], device=Q.device)
            .long()[:, None, None]
            .expand(*nbhd_idx.shape)
        )
        # Extract the points for each nbhd
        Q = Q[B, nbhd_idx]

        # Concat and return
        return self.fc_o(torch.cat([K, Q], dim=-1))


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
        A_ = Q.bmm(K.transpose(1, 2)) / math.sqrt(self.head_dim)

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


class RelativePositionKernel(nn.Module):
    def __init__(
        self, embed_dim, feature_dim, position_dim, n_heads, bias=False, lamda=1.0
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.feature_dim = feature_dim
        self.position_dim = position_dim
        self.lamda = lamda

        self.head_dim = embed_dim // n_heads
        assert (
            self.head_dim * self.n_heads == self.embed_dim
        ), "embed_dim must be divisible by n_heads"

        self.W_q = nn.Linear(feature_dim, embed_dim, bias=bias)
        self.W_k = nn.Linear(feature_dim, embed_dim, bias=bias)
        self.W_l = nn.Linear(position_dim, embed_dim, bias=bias)

        self.u = nn.Parameter(
            torch.zeros([self.n_heads, 1, self.head_dim], requires_grad=True)
        )
        self.v = nn.Parameter(
            torch.zeros([self.n_heads, 1, self.head_dim], requires_grad=True)
        )

    def forward(self, pairwise_locations, mask, query_features, key_features, nbhd_idx):

        # (bs, m, c_in) -> (bs, m, embed_dim) -> (bs * n_heads, m, h_dim)
        K = rearrange(self.W_k(key_features), "b n (h d) -> (b h) n d", h=self.n_heads)
        # (bs, n, c_in) -> (bs, n, embed_dim) -> (bs * n_heads, n, h_dim)
        Q = rearrange(
            self.W_q(query_features), "b n (h d) -> (b h) n d", h=self.n_heads
        )
        e = rearrange(
            self.W_l(pairwise_locations), "b n m (h d) -> (b h) n m d", h=self.n_heads
        )
        u = self.u.repeat([mask.shape[0], 1, 1])
        v = self.v.repeat([mask.shape[0], 1, 1])
        nbhd_idx = nbhd_idx.repeat_interleave(self.n_heads, dim=0)

        # Get NNS indexes
        NNS = (
            torch.arange(nbhd_idx.shape[1], device=nbhd_idx.device)[None, :, None]
            .long()
            .expand(*nbhd_idx.shape)
        )

        # Batch indicies
        B = (
            torch.arange(nbhd_idx.shape[0], device=nbhd_idx.device)[:, None, None]
            .long()
            .expand(*nbhd_idx.shape)
        )

        A_ = (
            Q.bmm(K.transpose(1, 2))[B, NNS, nbhd_idx]
            + self.lamda * (e @ (Q + v).unsqueeze(-1)).squeeze()
            + (u @ K.transpose(1, 2))[B, 0, nbhd_idx]
        ) / math.sqrt(self.head_dim)

        return rearrange(A_, "(b h) n m -> b n m h", h=self.n_heads)
