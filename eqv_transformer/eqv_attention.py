import sys

import torch
import torch.nn as nn
import torch.functional as F

from eqv_transformer.utils import Expression, export, Named, LinearBNact, Pass


class ReshapeFinalDim(nn.Module):
    """ Reshapes the final dimension into multiple dimensions
    """

    def __init__(self, new_dims):
        super().__init__()
        self.new_dims = new_dims

    def forward(self, x):
        return x.unsqueeze(-1).reshape(*x.shape[:-1], *self.new_dims)


class GlobalPool(nn.Module):
    """computes values reduced over all spatial locations in the mask
        Adapted from https://github.com/mfinzi/LieConv/
    """

    def __init__(self, mean=False):
        super().__init__()
        self.mean = mean

    def forward(self, x):
        """x [xyz (bs,n,ns,d), vals (bs,n,ns,c), mask (bs,n,ns)]"""
        if len(x) == 2:
            return x[1].mean(dim=(1, 2))
        coords, vals, mask = x
        summed = torch.where(mask.unsqueeze(-1), vals, torch.zeros_like(vals)).sum(
            dim=(1, 2)
        )
        if self.mean:
            summed /= mask.sum(-1).unsqueeze(-1)
        return summed


def NerualNetKernel(in_dim, out_dim, hid_dim=32, act="ReLU", bn=True):
    # Repuropsed from https://github.com/mfinzi/LieConv/
    return nn.Sequential(
        *LinearBNact(in_dim, hid_dim, act=act, bn=bn),
        *LinearBNact(hid_dim, hid_dim, act=act, bn=bn),
        *LinearBNact(hid_dim, out_dim, act=act, bn=bn),
    )


class SumKernel(nn.Module):
    def __init__(self, y_dim, g_dim, out_dim, hid_dim=32, act="ReLU", bn=True):
        # TODO: add options for different g and y kernels
        super().__init__()
        self.k_y = NerualNetKernel(
            in_dim=2 * y_dim, out_dim=1, hid_dim=hid_dim, act=act, bn=bn
        )

        self.k_g = NerualNetKernel(
            in_dim=g_dim, out_dim=1, hid_dim=hid_dim, act=act, bn=bn
        )

    def forward(self, pairwise_g, coset_functions, mask):
        """Computes the forward pass of the attention kernel.
        
        Parameters
        ----------
        pairwise_g : torch.Tensor
            Tensor of shape (batch_size, num_points, num_points, num_samples, num_samples, g_embed_dim + q_embed_dim + q_embed_dim).
            Precomputed pairwise differences between elements of the cosets.
        coset_functions : torch.Tensor
            Tensor of shape (batch_size, num_points, num_samples, y_dim).
            Represents the values of the functions over cosets at a given point.
        mask : torch.Tensor
            Tensor of dims (batch_size, num_points, coset_samples).
            Mask of points not included in each batch.
        
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_points, num_points, num_samples, num_samples)
        """

        # Expand the coset_functions's along the correct dimensions
        y_1 = (
            coset_functions.unsqueeze(1)
            .unsqueeze(3)
            .expand(*pairwise_g.shape[:-1], coset_functions.shape[-1])
        )
        y_2 = (
            coset_functions.unsqueeze(2)
            .unsqueeze(4)
            .expand(*pairwise_g.shape[:-1], coset_functions.shape[-1])
        )

        # Concatenate to get pairs of y values.
        y = torch.cat([y_1, y_2], dim=-1)

        # Compute the kernel elementwise on each point
        # TODO: may need to reshape the y and g before passing in.
        k = self.k_y(y) + self.k_g(pairwise_g)

        return k


@export
class Embed(nn.Module):
    """ Embeds a set of (x,y) points into the coset of x in group 
    attaches a function over that coset, with constant value of y.

    Parameters
    ----------
    group : Group
        The group of transformation of x to embed x to
    samples : int
        The number of samples from the coset to embed x with.
        If -1, for finite groups this will embed to the 
        whole coset.
    """

    def __init__(self, group, samples=-1):  #
        super().__init__()
        self.group = group()
        self.samples = samples

    def forward(self, input):
        """ Embed the inputs to the coset.
        
        Parameters
        ----------
        input : ([batch_size, points, x_dim], [batch_size, points, y_dim])
            Tuple containing arrays of x and y's to embed.
        
        returns : ([batch_size, points, stabiliser_samples, g_dim], [batch_size, points, stabiliser_samples, y_dim])
            Tuple containing: A representative element of x, Samples from an origin stabiliser, Function values at the origin stabiliser locations  
        """

        x, y, mask = input

        embedded_x, expanded_y, expanded_mask = self.group.lift(
            x, y, mask, samples=self.samples
        )

        return (embedded_x, expanded_y, expanded_mask)


@export
class EquivariantMultiheadAttention(nn.Module):
    def __init__(
        self, c_in, c_out, group, nbhd=32, act="Swish", bn=False, mlp_transforms=False
    ):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.group = group()

        self.kernels = [
            SumKernel(
                1,
                self.group.g_embed_dim + 2 * self.group.q_embed_dim,
                1,
                act=act,
                bn=bn,
            )
            for i in range(self.c_in)
        ]

        if mlp_transforms:
            self.output_transform = Pass(
                nn.Sequential(
                    *LinearBNact(self.c_in, self.c_out, act=act, bn=bn),
                    *LinearBNact(self.c_out, self.c_out, act=act, bn=bn),
                ),
                dim=1,
            )
        else:
            self.output_transform = Pass(
                nn.Linear(self.c_in, self.c_out, bias=False), dim=1
            )

    def forward(self, input):
        """ Forward pass for multihead equivariant attention.
        For each function, compute attention. Then combine them using an output matrix.
        
        Parameters
        ----------
        pairwise_embedded_x : torch.Tensor
            pairwise differences between embedded locations : torch.Tensor
            Shape (batch_size, num_points, num_points, coset_samples, coset_samples, g_embed_dim + q_embed_dim + q_embed_dim).
            Pairwise group element between the locations of points.
        coset_functions : torch.Tensor
            Location feature functions : torch.Tensor
            Shape (batch_size, num_points, coset_samples, num_functions). The feature
            functions associated with each point.
        mask : torch.Tensor
            Tensor of dims (batch_size, num_points, num_points, coset_samples, coset_samples).
            Mask of points not included in each batch.
        """

        pairwise_embedded_x, coset_functions, mask = input

        # TODO: various items commented
        # Subsample the points for efficiency
        # Extract the local neighbourhood only (cutoff kernel - could put in the kernel itself)

        # For EACH head
        coset_functions = list(torch.unbind(coset_functions, dim=-1))

        for i in range(self.c_in):
            # Compute the kernel between pairs - we could try tying these in some way to make the forward pass more efficient?
            #   e.g. one network with multiple heads for different attention heads.
            coset_functions[i] = coset_functions[i].unsqueeze(
                -1
            )  # add the last dimension back in - for futrue proofing
            pre_softmax_attention = self.kernels[i](
                pairwise_embedded_x, coset_functions[i], mask
            )

            # Softmax the kernels
            # Exponentiate
            softmax_attention = torch.exp(pre_softmax_attention)
            # Zero points not in the batch
            softmax_attention = torch.where(
                mask.unsqueeze(1)
                .unsqueeze(3)
                .unsqueeze(-1),  # expand mask along i and h_k axis
                softmax_attention,
                torch.zeros_like(softmax_attention),
            )
            # Compute the normalisation constant TODO: we could add the explicit monte carlo division here, and later, or ignore both.
            normalisation_const = softmax_attention.sum(
                dim=(2, 4), keepdim=True  # sum along the j, h_l axes
            )  # .expand(*softmax_attention.shape) # expand back to the correct shape for division
            # Normalise
            softmax_attention = softmax_attention / normalisation_const

            # Sum over the attention coefficients
            coset_functions[i] = coset_functions[i] + (
                softmax_attention
                * coset_functions[i]
                .unsqueeze(1)
                .unsqueeze(3)  # copy accross the i, h_k axes
                # .expand(
                #     *softmax_attention.shape
                # )  # expand the coset functions accross the
            ).sum(
                dim=(2, 4)
            )  # Sum over the j, h_l axes

        # Concatenate the heads
        coset_functions = torch.cat(coset_functions, dim=-1)

        # zero out the elements that are not in included in each batch - might not be needed
        coset_functions = coset_functions * mask.unsqueeze(-1)

        # Apply the output matrix to the functions
        output = (pairwise_embedded_x, coset_functions, mask)

        # Apply output transformation to the coset functions pointwise.
        output = self.output_transform(output)

        return output


class SimpleEqvTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_hidden,
        dim_output,
        n_enc_layers,
        n_dec_layers,
        num_outputs,
        group,
        coset_samples=100,
    ):
        super().__init__()

        self.dim_input = dim_input
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.num_outputs = num_outputs
        self.group = group

        enc_layers = []

        enc_layers.append(Embed(self.group, samples=coset_samples))
        enc_layers.append(
            EquivariantMultiheadAttention(dim_input, dim_hidden, group=self.group)
        )
        for _ in range(n_enc_layers - 1):
            enc_layers.append(
                EquivariantMultiheadAttention(dim_hidden, dim_hidden, group=self.group)
            )
        enc_layers.append(GlobalPool())

        self.enc = nn.Sequential(*enc_layers)

        dec_layers = []
        dec_layers.append(
            nn.Linear(dim_hidden, num_outputs * dim_output)
        )  # Linear mapping from pooled features to entities
        dec_layers.append(
            ReshapeFinalDim((num_outputs, dim_output))
        )  # Reshape correctly

        # TODO: This is a weak decoder, and does not employ "attention" pooling like Set transformers

        self.dec = nn.Sequential(*dec_layers)

    def forward(self, x, mask=None):

        # TODO: This is not general for all models. Hardcoded for constellations
        if mask is None:
            mask = torch.ones_like(x)

        y = (
            mask.unsqueeze(-1)
            .repeat((1,) * len(mask.shape) + (self.dim_input,))
            .float()
        )

        mask = mask.bool()

        return self.dec(self.enc((x, y, mask)))
