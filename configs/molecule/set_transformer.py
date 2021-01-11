from attrdict import AttrDict

import torch
from torch import nn
import torch.nn.functional as F

from lie_conv.datasets import SE3aug

from eqv_transformer.attention import SetTransformer
from eqv_transformer.molecule_predictor import MoleculePredictor

from forge import flags

flags.DEFINE_boolean(
    "data_augmentation", False, "Apply data augmentation to the input data or not"
)
flags.DEFINE_integer("n_enc_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("n_dec_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("num_heads", 4, "Number of attention heads.")
flags.DEFINE_integer(
    "n_inducing_points",
    0,
    "Number of inducing points; does not use inducing points if 0.",
)
flags.DEFINE_boolean("layer_norm", False, "Uses layer-norm if True.")
flags.DEFINE_integer("hidden_dim", 128, "Hidden dimension between layers")


class MolecueSetTransformer(SetTransformer):
    def __init__(self, num_species, charge_scale, aug=False, **kwargs):
        super().__init__(
            dim_input=3 + 3 * num_species, num_outputs=1, dim_output=1, **kwargs
        )
        self.charge_scale = charge_scale
        self.aug = aug
        self.random_rotate = SE3aug()

    # Featurization from lieconv
    def featurize(self, mb):
        charges = mb["charges"].float() / self.charge_scale.float()
        c_vec = torch.stack([torch.ones_like(charges), charges, charges ** 2], dim=-1)
        one_hot_charges = (
            (mb["one_hot"][:, :, :, None] * c_vec[:, :, None, :])
            .float()
            .reshape(*charges.shape, -1)
        )
        atomic_coords = mb["positions"].float()
        atom_mask = mb["charges"] > 0

        features = torch.cat([atomic_coords, one_hot_charges], dim=-1)

        return features, atom_mask.float()

    def forward(self, inpt):
        with torch.no_grad():
            if self.aug is not None:
                inpt["positions"], _, _ = self.random_rotate(
                    (inpt["positions"].float(), [], [])
                )
            features, mask = self.featurize(inpt)
        return super().forward(features, presence=mask)


def load(config, **unused_kwargs):
    del unused_kwargs

    encoder = MolecueSetTransformer(
        num_species=config.num_species,
        charge_scale=config.charge_scale,
        aug=config.data_augmentation,
        dim_hidden=config.hidden_dim,
        n_enc_layers=config.n_enc_layers,
        n_dec_layers=config.n_dec_layers,
        num_heads=config.n_heads,
        num_inducing_points=config.n_inducing_points,
        ln=config.layer_norm,
    )

    predictor = MoleculePredictor

    return (
        MoleculePredictor(encoder, task=config.task, ds_stats=config.ds_stats),
        f"SetTransformer",
    )
