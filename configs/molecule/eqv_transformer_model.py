import torch

from eqv_transformer.eqv_attention import EquivariantTransformer
from eqv_transformer.molecule_predictor import MoleculePredictor
from lie_conv.lieGroups import SE3, SO3, T, Trivial
from lie_conv.datasets import SE3aug

from forge import flags


flags.DEFINE_boolean(
    "data_augmentation",
    True,
    "Apply data augmentation to the data before passing to the model",
)
flags.DEFINE_integer("dim_hidden", 512, "Dimension of features to use in each layer")
flags.DEFINE_string(
    "activation_function", "swish", "Activation function to use in the network"
)
flags.DEFINE_boolean(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_heads", 8, "Number of attention heads in each layer")
flags.DEFINE_string(
    "block_norm",
    "layer_pre",
    "Type of norm to use in the attention block. none/[layer/batch]_[pre/post]",
)
flags.DEFINE_string(
    "output_norm",
    "none",
    "Type of norm to use in the final MLP layers block. none/layer/batch",
)
flags.DEFINE_string(
    "kernel_norm", "none", "The type of norm to use in the location kernels. none/batch"
)
flags.DEFINE_string(
    "kernel_type",
    "mlp",
    "Selects the type of attention kernel to use. mlp/relative_position/dot_product are valid",
)
flags.DEFINE_integer("kernel_dim", 16, "Hidden layer size to use in kernel MLPs")
flags.DEFINE_integer("num_layers", 6, "Number of ResNet layers to use")
flags.DEFINE_string("group", "SE3", "Group to be invariant to")
flags.DEFINE_integer(
    "lift_samples", 1, "Number of coset lift samples to use for non-trivial stabilisers"
)
flags.DEFINE_integer(
    "mc_samples",
    0,
    "Number of samples to use for estimating attention. 0 sets to use all points",
)
flags.DEFINE_float(
    "fill", 1.0, "Select mc_samples from K nearest mc_samples/fill points"
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")
flags.DEFINE_string(
    "architecture", "model_1", "The model architecture to use. model_1/lieconv"
)
flags.DEFINE_string(
    "attention_fn", "softmax", "Type of attention function to use. softmax/dot_product"
)
flags.DEFINE_integer(
    "feature_embed_dim",
    None,
    "Dimensionality of the embedding of the features for each head. Only used by some kernels",
)
flags.DEFINE_float(
    "max_sample_norm",
    None,
    "Maximum sample norm to allow through the lifting stage to prevent numerical issues.",
)
flags.DEFINE_string(
    "lie_algebra_nonlinearity",
    None,
    "Nonlinearity to apply to the norm of the lie algebra elements. Supported are None/tanh",
)


class MoleculeEquivariantTransformer(EquivariantTransformer):
    def __init__(self, num_species, charge_scale, aug=False, group=SE3, **kwargs):
        super().__init__(dim_input=3 * num_species, dim_output=1, group=group, **kwargs)
        self.charge_scale = charge_scale
        self.aug = aug
        self.random_rotate = SE3aug()

    def featurize(self, mb):
        charges = mb["charges"].to(mb["A"].dtype) / self.charge_scale
        c_vec = torch.stack(
            [torch.ones_like(charges), charges, charges ** 2], dim=-1
        )  #
        one_hot_charges = (
            (mb["one_hot"][:, :, :, None] * c_vec[:, :, None, :])
            .to(mb["A"].dtype)
            .reshape(*charges.shape, -1)
        )
        atomic_coords = mb["positions"].to(mb["A"].dtype)
        atom_mask = mb["charges"] > 0
        # print('orig_mask',atom_mask[0].sum())
        return (atomic_coords, one_hot_charges, atom_mask)

    def forward(self, mb):
        with torch.no_grad():
            x = self.featurize(mb)
            # x = mb
            x = self.random_rotate(x) if self.aug else x

        return super().forward(x).squeeze(-1)


def load(config, **unused_kwargs):

    if config.group == "SE3":
        group = SE3(0.2)
    elif config.group == "SO3":
        group = SO3(0.2)
    elif config.group == "T3":
        group = T(3)
    elif config.group == "Trivial3":
        group = Trivial(3)
    else:
        raise ValueError(f"{config.group} is and invalid group")

    torch.manual_seed(config.model_seed)
    predictor = MoleculeEquivariantTransformer(
        config.num_species,
        config.charge_scale,
        architecture=config.architecture,
        group=group,
        aug=config.data_augmentation,
        dim_hidden=config.dim_hidden,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        global_pool=True,
        global_pool_mean=config.mean_pooling,
        liftsamples=config.lift_samples,
        block_norm=config.block_norm,
        output_norm=config.output_norm,
        kernel_norm=config.kernel_norm,
        kernel_type=config.kernel_type,
        kernel_dim=config.kernel_dim,
        kernel_act=config.activation_function,
        fill=config.fill,
        mc_samples=config.mc_samples,
        attention_fn=config.attention_fn,
        feature_embed_dim=config.feature_embed_dim,
        max_sample_norm=config.max_sample_norm,
        lie_algebra_nonlinearity=config.lie_algebra_nonlinearity,
    )

    # predictor.net[-1][-1].weight.data = predictor.net[-1][-1].weight * (0.205 / 0.005)
    # predictor.net[-1][-1].bias.data = predictor.net[-1][-1].bias - (0.196 + 0.40)

    molecule_predictor = MoleculePredictor(predictor, config.task, config.ds_stats)

    return (
        molecule_predictor,
        f"MoleculeEquivariantTransformer_{config.group}_{config.architecture}",
    )
