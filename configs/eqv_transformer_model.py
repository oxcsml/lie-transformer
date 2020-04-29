from attrdict import AttrDict

import torch
from torch import nn
import torch.nn.functional as F
from eqv_transformer.classfier import Classifier
from eqv_transformer.eqv_attention import SimpleEqvTransformer
from eqv_transformer.groups.lie_groups import SE2

from forge import flags

flags.DEFINE_integer("input_dim", 2, "Dimensionality of the input.")
flags.DEFINE_integer("n_outputs", 4, "Number of output vectors.")
flags.DEFINE_integer("output_dim", 3, "Dimensionality of the output.")
flags.DEFINE_integer("n_enc_layers", 1, "Number of encoder layers.")
flags.DEFINE_integer("n_dec_layers", 0, "Number of encoder layers.")
# flags.DEFINE_integer("n_heads", 4, "Number of attention heads.")
# flags.DEFINE_integer(
#     "n_inducing_points",
#     0,
#     "Number of inducing points; does not use inducing points if 0.",
# )
# flags.DEFINE_boolean("layer_norm", False, "Uses layer-norm if True.")
flags.DEFINE_boolean("linear_transforms", True, "Add MLPs between attention")
flags.DEFINE_integer("coset_samples", 10, "Number of samples to take from the coset")


def load(config, **unused_kwargs):
    del unused_kwargs

    encoder = SimpleEqvTransformer(
        dim_input=config.input_dim,
        dim_hidden=64,
        dim_output=config.output_dim,
        num_outputs=config.n_outputs,
        n_enc_layers=config.n_enc_layers,
        n_dec_layers=config.n_dec_layers,
        group=SE2,
        coset_samples=config.coset_samples,
    )

    classifier = Classifier(encoder)
    return classifier
