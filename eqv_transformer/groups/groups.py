
import torch

# import numpy as np
from eqv_transformer.utils import export, Named


class Group(metaclass=Named):
    """Abstract class that defines the simple operations of a group that we require
    in order to be able to utilise it in our method.

    Inspiration from https://github.com/mfinzi/LieConv
    """

    # TODO: Support different spaces per group
    x_dim = NotImplemented  # dimension of the space of the group on which the group acts (possibly homogenous)

    g_embed_dim = (
        NotImplemented  # dimension of the representation of elements of the group.
    )
    # TODO: Is this the most sensible way?

    q_embed_dim = NotImplemented  # used for situations where the group is NOT transitive, but we have some Q = X / G. Dimension of the orbit embedding.

    def __init__(self):
        super().__init__()
        pass

    def lifted_elements(self, x, mask, samples):
        """ Lift elements from the space G acts on to G itself
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor of dims (batch_size, num_points, x_dim).
        mask : torch.Tensor
            Tensor of dims (batch_size, num_points).
            Mask of points not included in each batch.
        samples : Int
            The number of samples of the coset to take when embedding. If -1 for discrete groups,
            will sample the whole coset.

        Returns
        -------
        torch.Tensor
            Tensor of size (batch_size, num_points, num_samples, g_embed_dim)
                Tensor containing the embedding of samples from the coset of x.
        torch.Tensor
            Tensor of size (batch_size, num_points, num_samples, q_embed_dim)
                Tensor containing the orbit embeddings. None if q_embed_dim is 0.
        """
        raise NotImplementedError

    def inv(self, embedded_g):
        """ Computes the inverse of embedded group elements. This should be faster than inverting a 
        matrix form of the group elements.
        
        Parameters
        ----------
        g : torch.Tensor
            Tensor of dimensions (..., g_embed_dim). Group elements to invert

        Returns
        -------
        torch.Tensor
            Tensor of dimensions (..., g_embed_dim). 
            Inverse elements of the representations of G.
        """
        raise NotImplementedError

    def inv_matrix(self, matrix_g):
        """Computes matrix action inverses quickly for the specific group.
        By default computes the vector representation, inverts it, and converts
        back to matrix form.
        
        Parameters
        ----------
        matrix_g : torch.Tensor
            A tensor of shape (..., x_dim, x_dim).
        
        Returns
        -------
        torch.Tensor
            A tensor of shape (..., x_dim, x_dim). The inverse of matrix_g.
        """

        return self.representation_to_action(
            self.inv(self.action_to_representation(matrix_g))
        )

    # def self_act(self, embedded_g1, embedded_g2):
    #     """ Takes two tensors of vector representations of g and acts the first on the second

    #     Parameters
    #     ----------
    #     embedded_g1 : torch.Tensor
    #         Tensor of shape (..., g_embed_dim)
    #     embedded_g2 : torch.Tensor
    #         Tensor of shape (..., g_embed_dim)

    #     Returns
    #     -------
    #     torch.Tensor
    #         Tensor of dimensions (..., g_embed_dim).
    #         Vector representation of g_1 g_2.
    #     """

    def representation_to_action(self, embedded_g):
        """Computes a matrix version of the embedding of g that acts on a space
        of dimension x_dim from a g_embed_dim dimension vector representation of g.
        
        Parameters
        ----------
        embedded_g : torch.Tensor
            Tensor of size (..., g_embed_dim)
        
        Returns
        -------
        torch.Tensor
            Tensor of size (..., x_dim, x_dim)
        """

        raise NotImplementedError

    def action_to_representation(self, action):
        """Computes a g_embed_dim dimension vector representation of g from
        a x_dim matrix representation of g.
        
        Parameters
        ----------
        embedded_g : torch.Tensor
            Tensor of size (..., x_dim, x_dim)
        
        Returns
        -------
        torch.Tensor
            Tensor of size (..., g_embed_dim)
        """

        raise NotImplementedError

    def lift(self, x, y, mask, samples=-1):
        """ Lifts elements from x to samples from (or the whole of) the coset, and precomputes the pairwise
        distance between each point on each coset.
                
        Parameters
        ----------
        x : torch.Tensor 
            Tensor of shape (batch_size, num_points, x_dim).
            The points to embed.
        values : torch.Tensor
            Tensor of shape (batch_size, num_points, y_dim).
            The values associated with each point.
        mask : torch.Tensor
            Tensor of shape (batch_size, num_points, x_dim).
            Mask of points not included in each batch.
        samples : Int
            The number of samples of the coset to take when embedding. If -1 for discrete groups,
            will sample the whole coset.


        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_points, num_points, num_samples, num_samples, g_embed_dim + q_embed_dim + q_embed_dim)
            Tensor with the precomputed distances between each pair of group elements
        torch.Tensor
            Tensor of shape (batch_size, num_points, num_samples, y_dim)
            Tensor containing the y values copied correctly to the coset samples
        torch.Tensor
            Tensor of shape (batch_size, num_points, num_samples)
            Tensor containing the expanded mask of points not included in each batch.
        """
        # Compute the embedding of the point as a coset sample and an orbit embedding
        embedded_g, embedded_q = self.lifted_elements(
            x, mask, samples
        )  # (bs, n, d_x), (bs, n) -> (bs, n, ns, d_g), (bs, n, ns, d_q)

        # Expand the values and mask as appropriate
        expanded_y = y[..., None, :].repeat(
            (1,) * len(y.shape[:-1]) + (samples, 1)
        )  # (bs, n, d_y) -> (bs, n, ns, d_x)

        expanded_mask = mask[..., None].repeat(
            (1,) * len(mask.shape) + (samples,)
        )  # (bs, n) -> (bs, n, ns)

        # Convert the embeddings into pairwise differences
        pairwise_g = self.pairwise_difference(embedded_g)

        # Add the orbit embeddings if they exist
        if embedded_q is not None:
            q_1 = (
                embedded_q.unsqueeze(-3).unsqueeze(-2).expand(*pairwise_g.shape[:-1], 1)
            )
            q_2 = (
                embedded_q.unsqueeze(-4).unsqueeze(-3).expand(*pairwise_g.shape[:-1], 1)
            )
            embedded_x = torch.cat([pairwise_g, q_1, q_2], dim=-1)
        else:
            embedded_x = pairwise_g

        return embedded_x, expanded_y, expanded_mask

    def pairwise_difference(self, embedded_g):
        """ Computes the pairwise differences between elements of g
        Parameters
        ----------
        embedded_g : torch.Tensor
            Tensor of shape (batch_size, num_points, num_samples, g_embed_dim)
        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_points, num_points, num_samples, num_samples, g_embed_dim)
        """

        vinv = self.representation_to_action(
            self.inv(embedded_g).unsqueeze(-3).unsqueeze(-2)
        )
        u = self.representation_to_action(embedded_g.unsqueeze(-4).unsqueeze(-3))

        return self.action_to_representation(vinv @ u)

    def __str__(self):
        return f"{self.__class__}"

    def __repr__(self):
        return str(self)
