import collections
import functools
import numpy as np

import torch
from torchvision import transforms

from forge import flags

flags.DEFINE_integer('train_size', 100000, 'Number of training examples per epoch.')
flags.DEFINE_integer('test_size', 10000, 'Number of testing examples per epoch.')
flags.DEFINE_float('corner_noise', .1, 'See `create_constellations`.')
flags.DEFINE_boolean('shuffle_corners', True, 'See `create_constellations`.')

flags.DEFINE_float('pattern_upscale', 0., 'See `create_constellations`.')
flags.DEFINE_float('max_rotation_train', .33, 'See `create_constellations`.')
flags.DEFINE_float('pattern_drop_prob', .5, 'See `create_constellations`.')
flags.DEFINE_string('patterns_train', 'square,square,triangle,triangle,pentagon,pentagon,L,L', 'See `create_constellations`.')

flags.DEFINE_float('max_rotation_test', .33, 'See `create_constellations`.')
flags.DEFINE_string('patterns_test', 'square,square,triangle,triangle,pentagon,pentagon,L,L', 'See `create_constellations`.')

PATTERNS = {
    'square': [[1 + 2, 1 + 2, 1], [1, 1, 1], [1 + 2, 1, 1], [1, 1 + 2, 1]],
    'triangle': [[1, 1 + 1, 1], [1 + 2, 1, 1], [1 + 2, 1 + 2, 1]],
    'pentagon': [[1, 1, 1], [1 + 1, 1 - 2, 1], [1 + 2, 1 - 1, 1], [1 + 2, 1 + 1, 1], [1 + 1, 1 + 2, 1]],
    'L': [[1, 1, 1], [1 + 1, 1, 1], [1 + 2, 1, 1], [1 + 2, 1 + 1, 1], [1 + 2, 1 + 2, 1]],
}
PATTERNS = {k: np.asarray(v) for k, v in PATTERNS.items()}


def load(config, **unused_kwargs):
    del unused_kwargs

    gen_func_train = functools.partial(create_constellations,
                                 shuffle_corners=config.shuffle_corners,
                                 gaussian_noise=config.corner_noise,
                                 max_rot=config.max_rotation_train,
                                 max_upscale=config.pattern_upscale,
                                 drop_prob=config.pattern_drop_prob,
                                 which_patterns=config.patterns_train.split(','),
                                 )



    trainset = IteratorWrapper(gen_func_train, epoch_size=config.train_size, transform=lambda x: torch.tensor(x),
                               keys=['corners', 'presence', 'pattern_class_count'])

    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=config.batch_size, shuffle=False,
        num_workers=8, pin_memory=True)

    gen_func_test = functools.partial(create_constellations,
                                 shuffle_corners=config.shuffle_corners,
                                 gaussian_noise=config.corner_noise,
                                 max_rot=config.max_rotation_test,
                                 max_upscale=config.pattern_upscale,
                                 drop_prob=config.pattern_drop_prob,
                                 which_patterns=config.patterns_test.split(','),
                                 )

    # if config.test_size == config.train_size:
        # test_loader = train_loader
    #else:
    testset = IteratorWrapper(gen_func_test, epoch_size=config.test_size, transform=lambda x: torch.tensor(x),
                                  keys=['corners', 'presence', 'pattern_class_count'])

    test_loader = torch.utils.data.DataLoader(
            testset, batch_size=config.batch_size, shuffle=False,
            num_workers=8, pin_memory=True)

    return train_loader, test_loader


class IteratorWrapper():

    def __init__(self, gen_func, epoch_size, transform=None, keys=None, device=None):
        self.gen_func = gen_func
        self.epoch_size = epoch_size

        if device is not None:
            def device_transform(x):
                return x.to(device)

            transform = transforms.Compose([transform, device_transform])

        self.transform = transform
        self.keys = keys

    def __getitem__(self, item):
        if item >= self.epoch_size:
            raise StopIteration

        res = self.gen_func()
        if self.keys is not None:
            res = [res[k] for k in self.keys]
        else:
            res = list(res.values())

        if self.transform is not None:
            res = [self.transform(v) for v in res]

        res = [r.squeeze(0) for r in res]
        return res

    def __len__(self):
        return self.epoch_size


def create_constellations(
        size_n=1,
        shuffle_corners=True,
        gaussian_noise=0.,
        max_translation=1.,
        max_rot=0.0,
        max_upscale=0.0,
        which_patterns='all',
        drop_prob=0.0,
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

    if which_patterns == 'basic':
        which_patterns = 'square triangle'.split()

    elif which_patterns == 'all':
        which_patterns = list(PATTERNS.keys())

    elif isinstance(which_patterns, str):
        if which_patterns in PATTERNS:
            which_patterns = [which_patterns]
        else:
            raise ValueError('Pattern "{}" has not been '
                             'implemented.'.format(which_patterns))

    caps_dim = list(PATTERNS.values())[0].shape[1]
    transformations = []
    all_corners = []
    all_corner_presence = []
    all_pattern_presence = []

    centers = np.array([0, 0, 1])
    for i in range(len(which_patterns)):
        corners = centers.copy()

        corner_trans = np.zeros((PATTERNS[which_patterns[i]].shape[0], caps_dim, caps_dim))

        corner_trans[:, -1, :] = PATTERNS[which_patterns[i]]
        corner_trans[:, :-1, :-1] = np.eye(caps_dim - 1)
        corners = np.matmul(corners, corner_trans)
        corners = corners.reshape(-1, caps_dim)

        transformation = np.zeros((size_n, caps_dim, caps_dim))
        transformation[:, :, -1] = [0, 0, 1]

        # [pi/2, pi]
        degree = (np.random.random((size_n)) - .5) * 2. * np.pi * max_rot
        scale = 1. + np.random.random((size_n)) * max_upscale
        translation = np.random.random((size_n, 2)) * 24. * max_translation
        transformation[:, 0, 0] = np.cos(degree) * scale
        transformation[:, 1, 1] = np.cos(degree) * scale
        transformation[:, 0, 1] = np.sin(degree) * scale
        transformation[:, 1, 0] = -np.sin(degree) * scale
        transformation[:, -1, :-1] = translation / scale[Ellipsis, np.newaxis]

        corners = np.matmul(corners, transformation)

        random_pattern_choice = np.random.binomial(1, 1. - drop_prob,
                                                   (corners.shape[0], 1))

        random_corner_choice = np.tile(random_pattern_choice, (1, corners.shape[1]))

        all_corner_presence.append(random_corner_choice)
        all_pattern_presence.append(random_pattern_choice)
        transformations.append(transformation)
        all_corners.append(corners)

    capsules = np.concatenate(all_corners, axis=1)[Ellipsis, :2]
    all_corner_presence = np.concatenate(all_corner_presence, axis=1)
    all_pattern_presence = np.concatenate(all_pattern_presence, axis=1)

    pattern_ids = []
    current_pattern_id = 0
    pattern_class_count = collections.defaultdict(lambda: 0.)
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
            p = np.random.permutation(len(capsules[i]))
            capsules[i] = capsules[i][p]
            all_corner_presence[i] = all_corner_presence[i][p]
            pattern_ids[i] = pattern_ids[i][p]

    if gaussian_noise > 0.:
        capsules += np.random.normal(scale=gaussian_noise, size=capsules.shape)

    # normalize corners
    min_d, max_d = capsules.min(), capsules.max()
    capsules = (capsules - min_d) / (max_d - min_d + 1e-8) * 2 - 1.

    minibatch = dict(corners=capsules, presence=all_corner_presence,
                     pattern_presence=all_pattern_presence,
                     pattern_id=pattern_ids,
                     pattern_class_count=pattern_class_count)

    return minibatch
