from attrdict import AttrDict

import torch
from torch import nn
import torch.nn.functional as F
from eqv_transformer.classfier import Classifier
from eqv_transformer.attention import SetTransformer

from forge import flags

flags.DEFINE_integer("input_dim", 2, "Dimensionality of the input.")
flags.DEFINE_integer("n_outputs", 4, "Number of output vectors.")
flags.DEFINE_integer("output_dim", 3, "Dimensionality of the output.")
flags.DEFINE_integer("n_enc_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("n_dec_layers", 4, "Number of encoder layers.")
flags.DEFINE_integer("num_heads", 4, "Number of attention heads.")
flags.DEFINE_integer(
    "n_inducing_points",
    0,
    "Number of inducing points; does not use inducing points if 0.",
)
flags.DEFINE_boolean("layer_norm", False, "Uses layer-norm if True.")


def load(config, **unused_kwargs):
    del unused_kwargs

    encoder = SetTransformer(
        config.input_dim,
        config.n_outputs,
        config.output_dim,
        n_enc_layers=config.n_enc_layers,
        n_dec_layers=config.n_dec_layers,
        num_heads=config.num_heads,
        num_inducing_points=config.n_inducing_points,
        ln=config.layer_norm,
    )

    classifier = Classifier(encoder)
    return classifier, "SetTransformer"
