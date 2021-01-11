"""
This version of constellation data generation is based on the previous one.
There are two important changes:
    1. The dataset is not "online", i.e. generated on the go. So the size is fixed. 
    2. The constellations shapes are created using roots of unity, for better visualization.
"""

import collections
import functools
import numpy as np
import tensorflow as tf
import torch
from torchvision import transforms

# import ipdb
from forge import flags
import pickle
import os
import json

flags.DEFINE_integer("train_size", 10000, "Number of training examples per epoch.")
flags.DEFINE_integer("test_size", 1000, "Number of testing examples per epoch.")
flags.DEFINE_integer("naug", 2, "Number of augmentation.")
flags.DEFINE_float("corner_noise", 0.1, "See `create_constellations`.")
flags.DEFINE_boolean("shuffle_corners", True, "See `create_constellations`.")

flags.DEFINE_float("pattern_upscale", 0.0, "See `create_constellations`.")
flags.DEFINE_float("max_rotation", 0.33, "See `create_constellations`.")
flags.DEFINE_float("global_rotation_angle", 0.0, "See `create_constellations`.")
flags.DEFINE_float("global_translation", 0.0, "See `create_constellations`.")
flags.DEFINE_float("pattern_drop_prob", 0.5, "See `create_constellations`.")
flags.DEFINE_integer("patterns_reps", 2, "See `create_constellations`.")
flags.DEFINE_integer("data_seed", 0, "Seed for data generation.")


def roots_of_unity(n):
    x_coors = np.cos(2 * np.pi / n * np.arange(n)[..., np.newaxis])
    y_coors = np.sin(2 * np.pi / n * np.arange(n)[..., np.newaxis])

    coors = np.concatenate([x_coors, y_coors, np.tile([[1]], (n, 1))], axis=1)

    return coors


PATTERNS = {
    "triangle": roots_of_unity(3) + 2,
    "square": roots_of_unity(4) + 2,
    "pentagon": roots_of_unity(5) + 2,
    "L": np.asarray(
        [[1, 1, 1], [1 + 1, 1, 1], [1 + 2, 1, 1], [1 + 2, 1 + 1, 1], [1 + 2, 1 + 2, 1]]
    ),
}


def patterns(patterns_reps):
    return ("square,triangle,pentagon,L," * patterns_reps)[:-1].split(",")


def create_constellations(
    size_n=1,
    shuffle_corners=True,
    gaussian_noise=0.0,
    max_translation=1.0,
    global_translation=0.0,
    global_rotation_angle=0.0,
    max_rot=0.0,
    max_upscale=0.0,
    which_patterns="all",
    drop_prob=0.0,
    rng=None,
    **unused_kwargs
):
    """Creates a batch of data using numpy.
    Args:
        size_n: int, number of constellations to create.
        shuffle_corners: boolean, randomly shuffle points if True.
        gaussian_noise: float, std of random Gaussian noise injected into points.
        max_translation: float, maximum amount of translation applied to points.
        max_rot: float in [0., 1.], maximum rotation applied to each pattern will be max_rot * 2pi
        max_upscale: float, if non-zero all patterns are randomly scaled by scale in [1, 1 + max_upscale]
        which_patterns: 'all', 'basic' or a list of strings with pattern names, see PATTERNS.
        drop_prob: float in [0. 1.], probability of dropping out a pattern.
    """

    if rng is None:
        rng = np.random

    if which_patterns == "basic":
        which_patterns = "square triangle".split()

    elif which_patterns == "all":
        which_patterns = list(PATTERNS.keys())

    elif isinstance(which_patterns, str):
        if which_patterns in PATTERNS:
            which_patterns = [which_patterns]
        else:
            raise ValueError(
                'Pattern "{}" has not been ' "implemented.".format(which_patterns)
            )

    caps_dim = list(PATTERNS.values())[0].shape[1]
    transformations = []
    all_corners = []
    all_corner_presence = []
    all_pattern_presence = []

    centers = np.array([0, 0, 1])
    for i in range(len(which_patterns)):
        corners = centers.copy()

        corner_trans = np.zeros(
            (PATTERNS[which_patterns[i]].shape[0], caps_dim, caps_dim)
        )

        corner_trans[:, -1, :] = PATTERNS[which_patterns[i]]
        corner_trans[:, :-1, :-1] = np.eye(caps_dim - 1)
        corners = np.matmul(corners, corner_trans)
        corners = corners.reshape(-1, caps_dim)

        transformation = np.zeros((size_n, caps_dim, caps_dim))
        transformation[:, :, -1] = [0, 0, 1]

        # [pi/2, pi]
        degree = (rng.random((size_n)) - 0.5) * 2.0 * np.pi * max_rot
        scale = 1.0 + rng.random((size_n)) * max_upscale
        translation = rng.random((size_n, 2)) * 24.0 * max_translation
        transformation[:, 0, 0] = np.cos(degree) * scale
        transformation[:, 1, 1] = np.cos(degree) * scale
        transformation[:, 0, 1] = np.sin(degree) * scale
        transformation[:, 1, 0] = -np.sin(degree) * scale
        transformation[:, -1, :-1] = translation / scale[Ellipsis, np.newaxis]

        corners = np.matmul(corners, transformation)

        random_pattern_choice = rng.binomial(1, 1.0 - drop_prob, (corners.shape[0], 1))

        random_corner_choice = np.tile(random_pattern_choice, (1, corners.shape[1]))

        all_corner_presence.append(random_corner_choice)
        all_pattern_presence.append(random_pattern_choice)
        transformations.append(transformation)
        all_corners.append(corners)

    # if all patterns not present, make a random one present
    if np.count_nonzero(all_pattern_presence) == 0:
        idx = rng.randint(len(all_pattern_presence))
        
        all_pattern_presence[idx] = np.ones_like(all_pattern_presence[idx])
        all_corner_presence[idx] = np.ones_like(all_corner_presence[idx])

    capsules = np.concatenate(all_corners, axis=1)[Ellipsis, :2]
    all_corner_presence = np.concatenate(all_corner_presence, axis=1)
    all_pattern_presence = np.concatenate(all_pattern_presence, axis=1)

    pattern_ids = []
    current_pattern_id = 0
    pattern_class_count = collections.defaultdict(lambda: 0.0)
    for i, pattern in enumerate(which_patterns):
        corner_ids = [current_pattern_id] * len(PATTERNS[pattern])
        pattern_ids.extend(corner_ids)
        current_pattern_id += 1
        pattern_class_count[pattern] += all_pattern_presence[:, i]

    pattern_ids = np.asarray(pattern_ids)[np.newaxis]
    pattern_ids = np.tile(pattern_ids, [capsules.shape[0], 1])
    pattern_class_count = np.stack(list(pattern_class_count.values()), 1)

    capsules, all_corner_presence, all_pattern_presence = [
        i.astype(np.float32)
        for i in (capsules, all_corner_presence, all_pattern_presence)
    ]

    if shuffle_corners:
        for i in range(capsules.shape[0]):
            p = rng.permutation(len(capsules[i]))
            capsules[i] = capsules[i][p]
            all_corner_presence[i] = all_corner_presence[i][p]
            pattern_ids[i] = pattern_ids[i][p]

    if gaussian_noise > 0.0:
        capsules += rng.normal(scale=gaussian_noise, size=capsules.shape)

    # normalize corners
    min_d, max_d = capsules.min(), capsules.max()
    capsules = (capsules - min_d) / (max_d - min_d + 1e-8) * 2 - 1.0
    if global_rotation_angle != 0.0:
        rotation = np.array([[np.cos(global_rotation_angle), -np.sin(global_rotation_angle)], [np.sin(global_rotation_angle), np.cos(global_rotation_angle)]])
        _, b, c = capsules.shape
        capsules = np.matmul(rotation, capsules.reshape(b, c).transpose()).transpose().reshape(1, b, c)
    capsules += global_translation 

    minibatch = dict(
        corners=capsules,
        presence=all_corner_presence,
        pattern_presence=all_pattern_presence,
        pattern_id=pattern_ids,
        pattern_class_count=pattern_class_count,
    )

    return minibatch


def create_dataset(gen_func, epoch_size, transform=None, keys=None, device=None):
    if device is not None:

        def device_transform(x):
            return x.to(device)

        transform = transforms.Compose([transform, device_transform])

    def create_one_datapoint():
        res = gen_func()
        if keys is not None:
            res = [res[k] for k in keys]
        else:
            res = list(res.values())

        if transform is not None:
            res = [transform(v) for v in res]

        res = [r.squeeze(0) for r in res]
        return res

    data_list = [create_one_datapoint() for _ in range(epoch_size)]

    return data_list


def load(config):
    train_path = os.path.join(
        config.data_dir,
        "constellation/train_{}_{}.pkl".format(config.train_size, config.patterns_reps),
    )
    test_path = os.path.join(
        config.data_dir,
        "constellation/test_{}_{}_{}.pkl".format(config.test_size, config.naug, config.patterns_reps),
    )

    if tf.io.gfile.exists(train_path):
        with tf.io.gfile.GFile(train_path, "rb") as f:
            trainset = pickle.load(f)

    else:
        gen_func = functools.partial(
            create_constellations,
            shuffle_corners=config.shuffle_corners,
            gaussian_noise=config.corner_noise,
            max_rot=config.max_rotation,
            max_upscale=config.pattern_upscale,
            drop_prob=config.pattern_drop_prob,
            which_patterns=patterns(config.patterns_reps),
            rng=np.random.RandomState(seed=config.data_seed),
        )

        trainset = create_dataset(
            gen_func,
            epoch_size=config.train_size,
            transform=lambda x: torch.tensor(x),
            keys=["corners", "presence", "pattern_class_count"],
        )
        
    if config.naug > = 2 and tf.io.gfile.exists(test_path):
        print('Successfully reloaded test set')
        print(test_path)
        with tf.io.gfile.GFile(test_path, "rb") as f:
            testset = pickle.load(f)
    else:
        gen_func = functools.partial(
            create_constellations,
            shuffle_corners=config.shuffle_corners,
            gaussian_noise=config.corner_noise,
            max_rot=config.max_rotation,
            max_upscale=config.pattern_upscale,
            drop_prob=config.pattern_drop_prob,
            which_patterns=patterns(config.patterns_reps),
            rng=np.random.RandomState(seed=config.data_seed),
        )
        testset = create_dataset(
            gen_func,
            epoch_size=config.test_size,
            transform=lambda x: torch.tensor(x),
            keys=["corners", "presence", "pattern_class_count"],
        )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        testset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, test_loader, "constellation"
