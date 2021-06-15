import torch

from eqv_transformer.eqv_attention import EquivariantTransformer
from lie_conv.dynamicsTrainer import HNet
from lie_conv.hamiltonian import HamiltonianDynamics
from lie_conv.lieGroups import T, SE2, SE2_SZ_implementation, SO2
from eqv_transformer.dynamics_predictor import DynamicsPredictor

from forge import flags

flags.DEFINE_string("group", "T(2)", "Group to be invariant to.")


flags.DEFINE_integer("dim_hidden", 256, "Dimension of features to use in each layer")
flags.DEFINE_string(
    "activation_function", "swish", "Activation function to use in the network"
)
flags.DEFINE_boolean(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_heads", 8, "Number of attention heads in each layer")
flags.DEFINE_integer("kernel_dim", 16, "Hidden layer size to use in kernel MLPs")
flags.DEFINE_integer("num_layers", 4, "Number of ResNet layers to use")
flags.DEFINE_integer(
    "lift_samples",
    1,
    "Number of coset lift samples to use for non-trivial stabilisers.",
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")
flags.DEFINE_string(
    "attention_fn", "dot_product", "How to form the attention weights from the 'logits'."
)

flags.DEFINE_string(
    "block_norm", "layer_pre", "Normalization to use around the attention blocks."
)
flags.DEFINE_string("output_norm", "none", "Normalization to use in final output MLP.")
flags.DEFINE_string("kernel_norm", "none", "Normalization to use in kernel MLP.")
flags.DEFINE_string("kernel_type", "mlp", "Attention kernel type.")
flags.DEFINE_string("architecture", "model_1", "Overall model architecture.")
flags.DEFINE_boolean(
    "model_with_dict",
    True,
    "Makes model output predictions directly instead of a dictionary."
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
    if config.group == "T(2)":
        group = T(2)
    elif config.group == "T(3)":
        group = T(3)
    elif config.group == "SE(2)":
        group = SE2()
    elif config.group == "SE(2)_SZ":
        group = SE2_SZ_implementation()
    elif config.group == "SO(2)":
        group = SO2()
    else:
        raise NotImplementedError(f"Group {config.group} is not implemented.")

    torch.manual_seed(config.model_seed)  # TODO: initialization seed
    network = DynamicsEquivariantTransformer(
        group=group,
        dim_input=config.sys_dim,
        dim_output=1,  # Potential term in Hamiltonian is scalar
        dim_hidden=config.dim_hidden,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        global_pool=True,
        global_pool_mean=config.mean_pooling,
        liftsamples=config.lift_samples,
        kernel_dim=config.kernel_dim,
        kernel_act=config.activation_function,
        block_norm=config.block_norm,
        output_norm=config.output_norm,
        kernel_norm=config.kernel_norm,
        kernel_type=config.kernel_type,
        architecture=config.architecture,
        attention_fn=config.attention_fn,
    )

    if config.data_config == "configs/dynamics/nbody_dynamics_data.py":
        task = "nbody"
    elif config.data_config == "configs/dynamics/spring_dynamics_data.py":
        task = "spring"

    dynamics_predictor = DynamicsPredictor(network, debug=config.debug, task=task, model_with_dict=config.model_with_dict)

    return dynamics_predictor, "EqvTransformer_Dynamics"
