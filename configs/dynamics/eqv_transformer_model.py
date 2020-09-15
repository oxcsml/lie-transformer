import torch

from eqv_transformer.eqv_attention import EquivariantTransformer
from lie_conv.dynamicsTrainer import HNet
from lie_conv.hamiltonian import HamiltonianDynamics
from lie_conv.lieGroups import T
from eqv_transformer.dynamics_predictor import DynamicsPredictor

from forge import flags

flags.DEFINE_string("group", "T(2)", "Group to be invariant to.")


flags.DEFINE_integer("dim_hidden", 256, "Dimension of features to use in each layer")
flags.DEFINE_string(
    "activation_function", "swish", "Activation function to use in the network"
)
flags.DEFINE_boolean("layer_norm", True, "Use layer norm in the layers")
flags.DEFINE_boolean(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_heads", 8, "Number of attention heads in each layer")
flags.DEFINE_integer("kernel_dim", 16, "Hidden layer size to use in kernel MLPs")
flags.DEFINE_boolean("batch_norm", False, "Use batch norm in the kernel MLPs")
flags.DEFINE_integer("num_layers", 4, "Number of ResNet layers to use")
flags.DEFINE_integer(
    "lift_samples",
    1,
    "Number of coset lift samples to use for non-trivial stabilisers.",
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")

flags.DEFINE_boolean(
    "location_attention", True, "Use location kernel for attention weights."
)
flags.DEFINE_string(
    "attention_fn", "softmax", "How to form the attention weights from the 'logits'."
)
flags.DEFINE_boolean(
    "batch_norm_att",
    False,
    "Use batch norm instead of layer norm in each attention layer",
)
flags.DEFINE_boolean(
    "batch_norm_final_mlp",
    True,
    "Use batch norm before non-linearities in the last MLP after pooling.",
)


class DynamicsEquivariantTransformer(EquivariantTransformer, HNet):
    def __init__(self, center=True, **kwargs):
        super().__init__(**kwargs)
        self.center = center
        self.nfe = 0

    def forward(self, t, z, sysP, wgrad=True):
        dynamics = HamiltonianDynamics(
            lambda t, z: self.compute_H(z, sysP), wgrad=wgrad
        )
        return dynamics(t, z)

    def compute_V(self, x):
        """Input is a canonical position variable and the system parameters,
        shapes (bs, n,d) and (bs,n,c)"""
        q, sys_params = x
        mask = ~torch.isnan(q[..., 0])
        if self.center:
            q = q - q.mean(1, keepdims=True)
        return super().forward((q, sys_params, mask)).squeeze(-1)


def load(config, **unused_kwargs):

    # print('HARD CODED A default false loc att. REMOVE')

    if config.group == "T(2)":
        group = T(2)
    else:
        raise NotImplementedError(f"Group {config.group} is not implemented.")

    torch.manual_seed(config.model_seed)  # TODO: initialization seed
    network = DynamicsEquivariantTransformer(
        group=group,
        dim_input=config.space_dim,
        dim_output=1,  # Potential term in Hamiltonian is scalar
        dim_hidden=config.dim_hidden,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        layer_norm=config.layer_norm,
        global_pool=True,
        global_pool_mean=config.mean_pooling,
        liftsamples=config.lift_samples,
        kernel_dim=config.kernel_dim,
        kernel_act=config.activation_function,
        batch_norm=config.batch_norm,
        batch_norm_att=config.batch_norm_att,
        batch_norm_final_mlp=config.batch_norm_final_mlp,
        location_attention=config.location_attention,
        attention_fn=config.attention_fn,
    )

    dynamics_predictor = DynamicsPredictor(network, debug=config.debug)

    print(list(network.parameters())[0][:2])

    return dynamics_predictor, "EqvTransformer_Dynamics"
