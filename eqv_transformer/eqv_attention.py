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
from eqv_transformer.kernels import (
    AttentionKernel,
    SumKernel,
    DotProductKernel,
    RelativePositionKernel,
)


"""
Code implementing the main parts of the LieTransformer. 
This code builds on code written for https://arxiv.org/abs/2002.12880, 
in particular the use of their group classes.
"""


class EquivairantMultiheadAttention(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        n_heads,
        group,
        kernel_type="mlp",
        kernel_dim=16,
        act="swish",
        bn=False,
        mc_samples=0,
        fill=1.0,
        attention_fn="softmax",
        feature_embed_dim=None,
    ):

        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.n_heads = n_heads
        self.group = group

        self.mc_samples = mc_samples
        self.fill = fill
        self.kernel_type = kernel_type

        if not (attention_fn in ["softmax", "dot_product", "norm_exp"]):
            raise NotImplementedError(f"{attention_fn} not implemented.")
        self.attention_fn = attention_fn

        if len(kernel_type) == 4:
            normalisation = ["none", "softmax", "dot_product"]
            self.attention_fn = normalisation[int(kernel_type[0])]

            location_feature_combination = ["none", "sum", "mlp", "multiply"]
            location_feature_combination = location_feature_combination[
                int(kernel_type[1])
            ]

            feature_featurisation = [
                "none",
                "dot_product",
                "linear_concat",
                "linear_concat_linear",
            ]
            feature_featurisation = feature_featurisation[int(kernel_type[2])]

            location_featurisation = ["none", "mlp", "none"]
            location_featurisation = location_featurisation[int(kernel_type[3])]

            self.kernel = AttentionKernel(
                c_in,
                group.lie_dim + 2 * group.q_dim,
                n_heads,
                feature_featurisation=feature_featurisation,
                location_featurisation=location_featurisation,
                location_feature_combination=location_feature_combination,
                hidden_dim=kernel_dim,
                feature_embed_dim=feature_embed_dim,
                activation=act,
            )

        elif kernel_type == "mlp":
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
                n_heads,
            )
        elif kernel_type == "relative_position":
            self.kernel = RelativePositionKernel(
                c_in,
                c_in,
                group.lie_dim + 2 * group.q_dim,
                n_heads=n_heads,
                bias=True,
                lamda=1.0,
            )
        elif kernel_type == "dot_product_only":
            self.kernel = SumKernel(
                lambda x: [
                    None,
                    torch.zeros(x[1].shape[:-1], device=x[2].device).unsqueeze(-1),
                    None,
                ],  # unsure what's going on here. Dims don't match so trying to fix it.
                DotProductKernel(c_in, c_in, c_in, n_heads=n_heads),
                n_heads,
            )
        elif kernel_type == "location_only":
            self.kernel = SumKernel(
                MultiheadWeightNet(
                    group.lie_dim + 2 * group.q_dim,
                    1,
                    n_heads,
                    hid_dim=kernel_dim,
                    act=act,
                    bn=bn,
                ),
                lambda x1, x2, x3: 0,
                n_heads,
            )
        else:
            raise ValueError(f"{kernel_type} is not a valid kernel type")

        self.input_linear = nn.Linear(c_in, c_out)
        self.output_linear = nn.Linear(c_out, c_out)

    def extract_neighbourhoods(self, input, query_indices=None):
        """Extracts which points each other point is to attend to based on distance, or graph structure


        Parameters
        ----------
        input : (pairwise_g, coset_functions, mask)
        """
        # TODO: Currently no down sampling in this step.

        pairwise_g, coset_functions, mask = input

        if query_indices is not None:
            raise NotImplementedError()
        else:
            coset_functions_at_query = coset_functions
            mask_at_query = mask
            pairwise_g_at_query = pairwise_g

        if self.mc_samples > 0:
            dists = self.group.distance(pairwise_g_at_query)
            dists = torch.where(
                mask[:, None, :].expand(*dists.shape),
                dists,
                1e8 * torch.ones_like(dists),
            )
            k = (
                coset_functions.shape[1]
                if not self.mc_samples
                else min(self.mc_samples, coset_functions.shape[1])
            )
            k_ball = (
                coset_functions.shape[1]
                if not self.mc_samples
                else min(int(self.mc_samples / self.fill), coset_functions.shape[1])
            )
            _, points_in_ball_indices = dists.topk(
                k=k_ball, dim=-1, largest=False, sorted=False
            )
            ball_indices = torch.randperm(k_ball)[:k]

            nbhd_idx = points_in_ball_indices[:, :, ball_indices]

        else:
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
        # nbhd_coset_functions = coset_functions[
        #     BS, nbhd_idx
        # ]  # (bs, n * ns, c_in) -> (bs, n * ns, nbhd_size, c_in)
        nbhd_mask = mask[BS, nbhd_idx]  # (bs, n * ns) -> (bs, n * ns, nbhd_size)

        # (bs, n * ns, nbhd_size, g_dim), (bs, n * ns, nbhd_size, c_in), (bs, n * ns, nbhd_size), (bs, n * ns, nbhd_size)
        return (
            nbhd_pairwise_g,
            None,
            nbhd_mask,
            nbhd_idx,
            BS,
            NNS,
        )  # TODO: last two are conveniences - is there an easier way to do this?

    def forward(self, input):

        # (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_in), (bs, n * ns)
        pairwise_g, coset_functions, mask = input
        bs, n, d = coset_functions.shape

        # (bs, n * ns, nbhd_size, g_dim), (bs, n * ns, nbhd_size, c_in), (bs, n * ns, nbhd_size), (bs, n * ns, nbhd_size)
        (
            nbhd_pairwise_g,
            nbhd_coset_functions,
            nbhd_mask,
            nbhd_idx,
            BS,
            NNS,
        ) = self.extract_neighbourhoods(input)

        # (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_in), (bs, n * ns, nbhd_size, c_in) -> (bs, n * ns, nbhd_size, h)
        presoftmax_weights = self.kernel(
            nbhd_pairwise_g, nbhd_mask, coset_functions, coset_functions, nbhd_idx
        )

        if self.attention_fn == "softmax":
            # Make masked areas very small attention weights
            presoftmax_weights = torch.where(
                # (bs, n * ns, nbhd_size) -> (bs, n * ns, nbhd_size, 1). Constant along head dim
                nbhd_mask.unsqueeze(-1),
                presoftmax_weights,
                torch.tensor(
                    -1e38,
                    dtype=presoftmax_weights.dtype,
                    device=presoftmax_weights.device,
                )
                * torch.ones_like(presoftmax_weights),
            )

            # Compute the normalised attention weights
            # (bs, n * ns, nbhd_size, h) -> (bs, n * ns, nbhd_size, h)
            attention_weights = F.softmax(presoftmax_weights, dim=2)

        elif self.attention_fn == "norm_exp":
            # Make masked areas very small attention weights
            presoftmax_weights = torch.where(
                # (bs, n * ns, nbhd_size) -> (bs, n * ns, nbhd_size, 1). Constant along head dim
                nbhd_mask.unsqueeze(-1),
                presoftmax_weights,
                torch.tensor(
                    -1e38,
                    dtype=presoftmax_weights.dtype,
                    device=presoftmax_weights.device,
                )
                * torch.ones_like(presoftmax_weights),
            )

            # Compute the normalised attention weights
            # (bs, n * ns, nbhd_size, h) -> (bs, n * ns, nbhd_size, h)
            attention_weights = presoftmax_weights.exp()
            normalization = nbhd_mask.unsqueeze(-1).sum(-2, keepdim=True)
            normalization = torch.clamp(normalization, min=1)
            attention_weights = attention_weights / normalization

        # From the non-local attention paper
        elif self.attention_fn == "dot_product":
            attention_weights = torch.where(
                # (bs, n * ns, nbhd_size) -> (bs, n * ns, nbhd_size, 1). Constant along head dim
                nbhd_mask.unsqueeze(-1),
                presoftmax_weights,
                torch.tensor(
                    0.0,
                    dtype=presoftmax_weights.dtype,
                    device=presoftmax_weights.device,
                )
                * torch.ones_like(presoftmax_weights),
            )

            normalization = nbhd_mask.unsqueeze(-1).sum(-2, keepdim=True)
            normalization = torch.clamp(normalization, min=1)

            # Compute the normalised attention weights
            # (bs, n * ns, nbhd_size, h) -> (bs, n * ns, nbhd_size, h)
            attention_weights = attention_weights / normalization

        # Pass the inputs through the value linear layer
        # (bs, n * ns, nbhd_size, c_in) -> (bs, n * ns, nbhd_size, c_out)
        coset_functions = self.input_linear(coset_functions)

        attention_weights_expanded = torch.zeros(
            (bs, n, n, self.n_heads),
            dtype=attention_weights.dtype,
            device=attention_weights.device,
        )

        # (bs, n, n, h) hopefully?
        attention_weights_expanded[BS, NNS, nbhd_idx] = attention_weights
        attention_weights_expanded = rearrange(
            attention_weights_expanded, "b n m h -> b h n m"
        )

        coset_functions = rearrange(
            coset_functions, "b m (h d) -> b h m d", h=self.n_heads
        )

        coset_functions = attention_weights_expanded.matmul(coset_functions)
        coset_functions = rearrange(coset_functions, "b h n d -> b n (h d)")

        coset_functions = self.output_linear(coset_functions)

        # ( (bs, n * ns, n * ns, g_dim), (bs, n * ns, c_out), (bs, n * ns) )
        return (pairwise_g, coset_functions, mask)


class EquivariantTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        n_heads,
        group,
        block_norm="layer_pre",
        kernel_norm="none",
        kernel_type="mlp",
        kernel_dim=16,
        kernel_act="swish",
        hidden_dim_factor=1,
        mc_samples=0,
        fill=1.0,
        attention_fn="softmax",
        feature_embed_dim=None,
    ):
        super().__init__()
        self.ema = EquivairantMultiheadAttention(
            dim,
            dim,
            n_heads,
            group,
            kernel_type=kernel_type,
            kernel_dim=kernel_dim,
            act=kernel_act,
            bn=kernel_norm == "batch",
            mc_samples=mc_samples,
            fill=fill,
            attention_fn=attention_fn,
            feature_embed_dim=feature_embed_dim,
        )

        self.mlp = MLP(dim, dim, dim, 2, kernel_act, kernel_norm == "batch")

        if block_norm == "none":
            self.attention_function = lambda inpt: inpt[1] + self.ema(inpt)[1]
            self.mlp_function = lambda inpt: inpt[1] + self.mlp(inpt)[1]
        elif block_norm == "layer_pre":
            self.ln_ema = nn.LayerNorm(dim)
            self.ln_mlp = nn.LayerNorm(dim)

            self.attention_function = (
                lambda inpt: inpt[1]
                + self.ema((inpt[0], self.ln_ema(inpt[1]), inpt[2]))[1]
            )
            self.mlp_function = (
                lambda inpt: inpt[1]
                + self.mlp((inpt[0], self.ln_mlp(inpt[1]), inpt[2]))[1]
            )
        elif block_norm == "layer_post":
            self.ln_ema = nn.LayerNorm(dim)
            self.ln_mlp = nn.LayerNorm(dim)

            self.attention_function = lambda inpt: inpt[1] + self.ln_ema(
                self.ema(inpt)[1]
            )
            self.mlp_function = lambda inpt: inpt[1] + self.ln_mlp(self.mlp(inpt)[1])
        elif block_norm == "batch_pre":
            self.bn_ema = MaskBatchNormNd(dim)
            self.bn_mlp = MaskBatchNormNd(dim)

            self.attention_function = (
                lambda inpt: inpt[1] + self.ema(self.bn_ema(inpt))[1]
            )
            self.mlp_function = lambda inpt: inpt[1] + self.mlp(self.bn_mlp(inpt))[1]
        elif block_norm == "batch_post":
            self.bn_ema = MaskBatchNormNd(dim)
            self.bn_mlp = MaskBatchNormNd(dim)

            self.attention_function = (
                lambda inpt: inpt[1] + self.bn_ema(self.ema(inpt))[1]
            )
            self.mlp_function = lambda inpt: inpt[1] + self.bn_mlp(self.mlp(inpt))[1]
        else:
            raise ValueError(f"{block_norm} is invalid block norm type.")

    def forward(self, inpt):
        inpt[1] = self.attention_function(inpt)
        inpt[1] = self.mlp_function(inpt)

        return inpt


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

        if self.mean:
            # mean pooling
            summed = torch.where(mask.unsqueeze(-1), vals, torch.zeros_like(vals)).sum(
                1
            )
            summed_mask = mask.sum(-1).unsqueeze(-1)
            summed_mask = torch.where(
                summed_mask == 0, torch.ones_like(summed_mask), summed_mask
            )
            summed /= summed_mask

            return summed
        else:
            # max pooling
            masked = torch.where(
                mask.unsqueeze(-1),
                vals,
                torch.tensor(
                    -1e38,
                    dtype=vals.dtype,
                    device=vals.device,
                )
                * torch.ones_like(vals),
            )

            return masked.max(dim=1)[0]


class EquivariantTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_output,
        dim_hidden,
        num_layers,
        num_heads,
        global_pool=True,
        global_pool_mean=True,
        group=SE3(0.2),
        liftsamples=1,
        block_norm="layer_pre",
        output_norm="none",
        kernel_norm="none",
        kernel_type="mlp",
        kernel_dim=16,
        kernel_act="swish",
        mc_samples=0,
        fill=1.0,
        architecture="model_1",
        attention_fn="softmax",  # softmax or dot product? SZ: TODO: "dot product" is used to describe both the attention weights being non-softmax (non-local attention paper) and the feature kernel. should fix terminology
        feature_embed_dim=None,
        max_sample_norm=None,
        lie_algebra_nonlinearity=None,
    ):
        super().__init__()

        if isinstance(dim_hidden, int):
            dim_hidden = [dim_hidden] * (num_layers + 1)

        if isinstance(num_heads, int):
            num_heads = [num_heads] * num_layers

        attention_block = lambda dim, n_head: EquivariantTransformerBlock(
            dim,
            n_head,
            group,
            block_norm=block_norm,
            kernel_norm=kernel_norm,
            kernel_type=kernel_type,
            kernel_dim=kernel_dim,
            kernel_act=kernel_act,
            mc_samples=mc_samples,
            fill=fill,
            attention_fn=attention_fn,
            feature_embed_dim=feature_embed_dim,
        )

        activation_fn = {
            "swish": Swish,
            "relu": nn.ReLU,
            "softplus": nn.Softplus,
        }

        if architecture == "model_1":
            if output_norm == "batch":
                norm1 = nn.BatchNorm1d(dim_hidden[-1])
                norm2 = nn.BatchNorm1d(dim_hidden[-1])
                norm3 = nn.BatchNorm1d(dim_hidden[-1])
            elif output_norm == "layer":
                norm1 = nn.LayerNorm(dim_hidden[-1])
                norm2 = nn.LayerNorm(dim_hidden[-1])
                norm3 = nn.LayerNorm(dim_hidden[-1])
            elif output_norm == "none":
                norm1 = nn.Sequential()
                norm2 = nn.Sequential()
                norm3 = nn.Sequential()
            else:
                raise ValueError(f"{output_norm} is not a valid norm type.")

            self.net = nn.Sequential(
                Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
                *[
                    attention_block(dim_hidden[i], num_heads[i])
                    for i in range(num_layers)
                ],
                GlobalPool(mean=global_pool_mean)
                if global_pool
                else Expression(lambda x: x[1]),
                nn.Sequential(
                    norm1,
                    activation_fn[kernel_act](),
                    nn.Linear(dim_hidden[-1], dim_hidden[-1]),
                    norm2,
                    activation_fn[kernel_act](),
                    nn.Linear(dim_hidden[-1], dim_hidden[-1]),
                    norm3,
                    activation_fn[kernel_act](),
                    nn.Linear(dim_hidden[-1], dim_output),
                ),
            )
        elif architecture == "lieconv":
            if output_norm == "batch":
                norm = nn.BatchNorm1d(dim_hidden[-1])
            elif output_norm == "none":
                norm = nn.Sequential()
            else:
                raise ValueError(f"{output_norm} is not a valid norm type.")

            self.net = nn.Sequential(
                Pass(nn.Linear(dim_input, dim_hidden[0]), dim=1),
                *[
                    attention_block(dim_hidden[i], num_heads[i])
                    for i in range(num_layers)
                ],
                nn.Sequential(
                    OrderedDict(
                        [
                            # ("norm", Pass(norm, dim=1)),
                            (
                                "activation",
                                Pass(
                                    activation_fn[kernel_act](),
                                    dim=1,
                                ),
                            ),
                            (
                                "linear",
                                Pass(nn.Linear(dim_hidden[-1], dim_output), dim=1),
                            ),
                        ]
                    )
                ),
                GlobalPool(mean=global_pool_mean)
                if global_pool
                else Expression(lambda x: x[1]),
            )
        else:
            raise ValueError(f"{architecture} is not a valid architecture.")

        self.group = group
        self.liftsamples = liftsamples
        self.max_sample_norm = max_sample_norm

        self.lie_algebra_nonlinearity = lie_algebra_nonlinearity
        if lie_algebra_nonlinearity is not None:
            if lie_algebra_nonlinearity == "tanh":
                self.lie_algebra_nonlinearity = nn.Tanh()
            else:
                raise ValueError(
                    f"{lie_algebra_nonlinearity} is not a supported nonlinearity"
                )

    def forward(self, input):
        if self.max_sample_norm is None:
            lifted_data = self.group.lift(input, self.liftsamples)
        else:
            lifted_data = [
                torch.tensor(self.max_sample_norm * 2, device=input[0].device),
                0,
                0,
            ]
            while lifted_data[0].norm(dim=-1).max() > self.max_sample_norm:
                lifted_data = self.group.lift(input, self.liftsamples)

        if self.lie_algebra_nonlinearity is not None:
            lifted_data = list(lifted_data)
            pairs_norm = lifted_data[0].norm(dim=-1) + 1e-6
            lifted_data[0] = lifted_data[0] * (
                self.lie_algebra_nonlinearity(pairs_norm / 7) / pairs_norm
            ).unsqueeze(-1)

        return self.net(lifted_data)
