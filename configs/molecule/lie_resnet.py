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
flags.DEFINE_integer(
    "nbhd_size", 25, "The number of samples to use for Monte Carlo estimation"
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
    "lift_samples", 4, "Number of coset lift samples to use for non-trivial stabilisers"
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")
flags.DEFINE_string(
    "lie_algebra_nonlinearity",
    None,
    "Nonlinearity to apply to the norm of the lie algebra elements. Supported are None/tanh",
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

    torch.manual_seed(config.model_seed)  # TODO: temp fix of seed
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
        lie_algebra_nonlinearity=config.lie_algebra_nonlinearity,
    )

    molecule_predictor = MoleculePredictor(predictor, config.task, config.ds_stats)

    return molecule_predictor, f"MoleculeLieResNet_{config.group}"
