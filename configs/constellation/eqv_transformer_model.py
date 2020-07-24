import torch

from eqv_transformer.classfier import Classifier
from eqv_transformer.eqv_attention import EquivariantTransformer
from lie_conv.lieGroups import SE3, SO3, T, Trivial
from lie_conv.datasets import SE3aug

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
flags.DEFINE_boolean("layer_norm", False, "Use layer norm in the layers")
flags.DEFINE_boolean(
    "mean_pooling",
    True,
    "Use mean pooling insteave of sum pooling in the invariant layer",
)
flags.DEFINE_integer("num_heads", 8, "Number of attention heads in each layer")
flags.DEFINE_integer("kernel_dim", 16, "Hidden layer size to use in kernel MLPs")
flags.DEFINE_boolean("batch_norm", False, "Use batch norm in the kernel MLPs")
flags.DEFINE_integer("num_layers", 6, "Number of ResNet layers to use")
flags.DEFINE_string("group", "SE3", "Group to be invariant to")
flags.DEFINE_integer(
    "lift_samples", 1, "Number of coset lift samples to use for non-trivial stabilisers"
)
flags.DEFINE_integer("model_seed", 0, "Model rng seed")

flags.DEFINE_string("content_type", "pairwise_distances", "How to initialize y")


def constant_features(X):
    return torch.ones(X.shape[:-1]).unsqueeze(-1)

def pairwise_distance_features(X):
    X_pairs = X.unsqueeze(2).transpose(1, 2) - X.unsqueeze(2)
    X_distances = torch.norm(X_pairs, dim=-1)
    X_distances = torch.sort(X_distances, dim=-1, descending=True)[0][..., :-1]
    return X_distances

def pairwise_distance_moment_features(X, n_moments=5):
    X_pairs = X.unsqueeze(2).transpose(1, 2) - X.unsqueeze(2)
    X_distances = torch.norm(X_pairs, dim=-1)
    X_distances = torch.sort(X_distances, dim=-1, descending=True)[0][..., :-1]
    X_distance_moments = torch.cat([
        X_distances.pow(i).mean(dim=-1, keepdim=True)
    ], dim=-1)
    return X_distance_moments


class ConstellationEquivariantTransformer(EquivariantTransformer):
    def __init__(self, content_type="pairwise_distances", **kwargs):
        if content_type == 'centroidal':
            dim_input = 2
            raise NotImplementedError()
            # self.featurize 
        elif content_type == 'constant':
            dim_input = 1
            self.featurize = constant_features
        elif content_type == 'pairwise_distances':
            self.featurize = pairwise_distance_features # i.e. use the arg dim_input
        elif content_type == 'distance_moments':
            dim_input = 5
            self.featurize = lambda X: pairwise_distance_moment_features(X, n_moments=dim_input)

        super().__init__(self, **kwargs, dim_input=dim_input)

    def forward(self, X, presence=None):
        with torch.no_grad():
            Y = self.featurize(mb)
        
        return super().forward((X, Y, presence))


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
    predictor = ConstellationEquivariantTransformer(
        content_type=config.content_type
        group=group,
        aug=config.data_augmentation,
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

    return molecule_predictor, f"ConstellationEquivariantTransformer_{config.group}"
