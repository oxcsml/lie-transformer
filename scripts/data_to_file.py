'''Generate constellation data and write data to a file'''
import sys
sys.path.append('forge')
sys.path.append('.')
import os
import functools
import pickle
import numpy as np
import forge
from forge import flags
import torch

from configs.constellation.constellation_v2 import create_constellations, create_dataset, patterns
import tensorflow as tf
config = forge.config()

def generate_examples(path):
    gen_func = functools.partial(create_constellations,
                             shuffle_corners=config.shuffle_corners,
                             gaussian_noise=config.corner_noise,
                             max_rot=config.max_rotation,
                             max_upscale=config.pattern_upscale,
                             drop_prob=config.pattern_drop_prob,
                             which_patterns=patterns(config.patterns_reps),
                             rng=np.random.RandomState(
                                 seed=config.data_seed)
                             )

    trainset = create_dataset(gen_func, epoch_size=config.train_size, transform=lambda x: torch.tensor(x),
                          keys=['corners', 'presence', 'pattern_class_count'])

    testset = create_dataset(gen_func, epoch_size=config.test_size, transform=lambda x: torch.tensor(x),
                         keys=['corners', 'presence', 'pattern_class_count'])

    train_path = os.path.join(path, 'train_{}_{}.pkl'.format(config.train_size, config.patterns_reps))
    test_path = os.path.join(path, 'test_{}_{}.pkl'.format(config.test_size, config.patterns_reps))

    with tf.gfile.GFile(train_path, 'w') as f:
       pickle.dump(trainset, f)

    with tf.gfile.GFile(test_path, 'w') as f:
        pickle.dump(testset, f)

if __name__ == '__main__':
    generate_examples('./data/constellation')


