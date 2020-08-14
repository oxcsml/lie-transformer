import torch

from eqv_transformer.classfier import Classifier
from eqv_transformer.eqv_attention import EquivariantTransformer
from lie_conv.lieGroups import SE3, SE2, SO3, T, Trivial
# from lie_conv.datasets import SE3aug

from forge import flags


flags.DEFINE_boolean(
    "data_augmentation",
    False,
    "Apply data augmentation to the data before passing to the model",
)
flags.DEFINE_integer("dim_hidden", 512, "Dimension of features to use in each layer")
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
flags.DEFINE_integer("num_layers", 6, "Number of ResNet layers to use")
flags.DEFINE_string("group", "SE2", "Group to be invariant to")
flags.DEFINE_integer(
    "lift_samples", 1, "Number of coset lift samples to use for non-trivial stabilisers"
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")

flags.DEFINE_string("content_type", "pairwise_distances", "How to initialize y")


def constant_features(X, presence):
    return torch.ones(X.shape[:-1]).unsqueeze(-1)


def pairwise_distance_features(X, presence):
    X_pairs = X.unsqueeze(2).transpose(1, 2) - X.unsqueeze(2)
    X_distances = torch.norm(X_pairs, dim=-1)
    X_distances = torch.where(
        presence.unsqueeze(-1), X_distances, torch.zeros_like(X_distances)
    )
    X_distances = torch.sort(X_distances, dim=-1, descending=True)[0][..., :-1]
    return X_distances


def pairwise_distance_moment_features(X, presence, n_moments=5):
    X_pairs = X.unsqueeze(2).transpose(1, 2) - X.unsqueeze(2)
    X_distances = torch.norm(X_pairs, dim=-1)
    X_distances = torch.where(
        presence.unsqueeze(-1), X_distances, torch.zeros_like(X_distances)
    )
    X_distance_moments = torch.cat(
        [X_distances.pow(i).sum(dim=-1, keepdim=True) for i in range(1, n_moments + 1)],
        dim=-1,
    )
    N = presence.sum(dim=-1, keepdim=True).unsqueeze(-1)
    N = torch.where(N == 0, torch.ones_like(N), N)
    X_distance_moments = X_distance_moments / N
    return X_distance_moments


class ConstellationEquivariantTransformer(EquivariantTransformer):
    def __init__(
        self, n_patterns, patterns_reps, content_type="pairwise_distances", **kwargs
    ):
        if content_type == "centroidal":
            dim_input = 2
        elif content_type == "constant":
            dim_input = 1
        elif content_type == "pairwise_distances":
            dim_input = patterns_reps * 17 - 1
        elif content_type == "distance_moments":
            dim_input = 10
        else:
            raise NotImplementedError(f"{content_type} featurization not implemented")

        super().__init__(
            dim_input=dim_input, dim_output=n_patterns * (patterns_reps + 1), **kwargs,
        )

        self.n_patterns = n_patterns
        self.patterns_reps = patterns_reps

        if content_type == "centroidal":
            raise NotImplementedError()
            # self.featurize
        elif content_type == "constant":
            self.featurize = constant_features
        elif content_type == "pairwise_distances":
            self.featurize = pairwise_distance_features  # i.e. use the arg dim_input
        elif content_type == "distance_moments":
            self.featurize = lambda X, presence: pairwise_distance_moment_features(
                X, presence, n_moments=dim_input
            )

    def forward(self, X, presence=None):
        presence = presence.to(bool)
        with torch.no_grad():
            Y = self.featurize(X, presence)

        output = super().forward((X, Y, presence.to(bool)))
        return output.reshape(
            [*output.shape[:-1], self.n_patterns, self.patterns_reps + 1]
        )


def load(config, **unused_kwargs):

    if config.group == "SE3":
        group = SE3(0.2)
    if config.group == "SE2":
        group = SE2(0.2)
    elif config.group == "SO3":
        group = SO3(0.2)
    elif config.group == "T3":
        group = T(3)
    elif config.group == "Trivial3":
        group = Trivial(3)
    else:
        raise ValueError(f"{config.group} is and invalid group")

    input_dim = config.patterns_reps * 17 - 1
    output_dim = config.patterns_reps + 1

    torch.manual_seed(config.model_seed)
    predictor = ConstellationEquivariantTransformer(
        n_patterns=4,
        patterns_reps=config.patterns_reps,
        content_type=config.content_type,
        group=group,
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
    )

    classifier = Classifier(predictor)

    return classifier, f"ConstellationEquivariantTransformer_{config.group}"
