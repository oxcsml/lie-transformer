import torch

from lie_conv.moleculeTrainer import MolecLieResNet

from eqv_transformer.molecule_predictor import MoleculePredictor
from lie_conv.lieGroups import SE3, SO3, T, Trivial

from forge import flags

flags.DEFINE_bool(
    "data_augmentation",
    False,
    "Apply data augmentation to the data before passing to the model",
)
flags.DEFINE_integer("channel_width", 1536, "Number of channels to use in each layer")
flags.DEFINE_integer(
    "nbhd_size", 100, "The number of samples to use for Monte Carlo estimation"
)
flags.DEFINE_string(
    "activation_function", "swish", "Activation function to use in the network"
)
flags.DEFINE_boolean("batch_norm", True, "Use batch norm in the layers")
flags.DEFINE_bool(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_layers", 6, "Number of ResNet layers to use")
flags.DEFINE_string("group", "SE3", "Group to be invariant to")
flags.DEFINE_integer("channels", 1536, "Number of channels in the conv layers")
flags.DEFINE_float(
    "fill",
    1.0,
    "specifies the fraction of the input which is included in local neighborhood. (can be array to specify a different value for each layer",
)
flags.DEFINE_integer(
    "lift_samples", 1, "Number of coset lift samples to use for non-trivial stabilisers"
)


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

    torch.manual_seed(0)  # TODO: temp fix of seed
    predictor = MolecLieResNet(
        config.num_species,
        config.charge_scale,
        group=group,
        aug=config.data_augmentation,
        k=config.channels,
        nbhd=config.nbhd_size,
        act=config.activation_function,
        bn=config.batch_norm,
        mean=config.mean_pooling,
        num_layers=config.num_layers,
        fill=config.fill,
        liftsamples=config.lift_samples,
    )

    molecule_predictor = MoleculePredictor(predictor, config.task, config.ds_stats)

    return molecule_predictor, f"MolecLieResNet_{config.group}"
