from lie_conv.moleculeTrainer import MolecLieResNet

from eqv_transformer.molecule_predictor import MoleculePredictor

from forge import flags

flags.DEFINE_bool(
    "data_augmantation",
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


def load(config, **unused_kwargs):
    predictor = MolecLieResNet(
        config.num_species,
        config.charge_scale,
        aug=config.data_augmentation,
        k=config.channels,
        nbhd=config.nbhd_size,
        act=config.activation_function,
        bn=config.batch_norm,
        mean=config.mean_pooling,
        num_layers=config.num_layers,
    )

