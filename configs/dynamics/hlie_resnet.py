import torch

from lie_conv.dynamicsTrainer import HLieResNet
from lie_conv.lieGroups import T
from eqv_transformer.dynamics_predictor import DynamicsPredictor

from forge import flags

flags.DEFINE_string("group", "T(2)", "Group to be invariant to.")
flags.DEFINE_integer("k", 384, "Channel width for the network.")
flags.DEFINE_integer("num_layers", 4, "Number of layers.")


def load(config, **unused_kwargs):

    if config.group == "T(2)":
        group = T(2)
    else:
        raise NotImplementedError(f"Group {config.group} is not implemented.")

    torch.manual_seed(0)  # TODO: initialization seed
    network = HLieResNet(
        sys_dim=config.sys_dim,
        d=config.space_dim,
        group=group,
        k=config.k,
        num_layers=config.num_layers,
    )

    dynamics_predictor = DynamicsPredictor(network)

    return dynamics_predictor, "HLieResNet_Dynamics"
