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
import ipdb
import math
from configs.constellation.constellation import create_constellations, create_dataset, patterns
import tensorflow as tf
config = forge.config()

def generate_train(path):
    gen_func = functools.partial(create_constellations,
                             shuffle_corners=config.shuffle_corners,
                             gaussian_noise=config.corner_noise,
                             max_rot=0,#config.max_rotation,
                             max_upscale=config.pattern_upscale,
                             drop_prob=config.pattern_drop_prob,
                             which_patterns=patterns(config.patterns_reps),
                             rng=np.random.RandomState(
                                 seed=1234)
                             )

    trainset = create_dataset(gen_func, epoch_size=config.train_size, transform=lambda x: torch.tensor(x),
                          keys=['corners', 'presence', 'pattern_class_count'])

    train_path = os.path.join(path, 'maxrot0train_{}_{}.pkl'.format(config.train_size, config.patterns_reps))

    with tf.gfile.GFile(train_path, 'w') as f:
       pickle.dump(trainset, f)

def generate_test(path):
    gen_func = functools.partial(create_constellations,
                             shuffle_corners=config.shuffle_corners,
                             gaussian_noise=config.corner_noise,
                             max_rot=0,#config.max_rotation,
                             max_upscale=config.pattern_upscale,
                             drop_prob=config.pattern_drop_prob,
                             which_patterns=patterns(config.patterns_reps),
                             rng=np.random.RandomState(
                                 seed=0)
                             )

    testset = create_dataset(gen_func, epoch_size=config.test_size, transform=lambda x: torch.tensor(x),
                         keys=['corners', 'presence', 'pattern_class_count'])

    test_path = os.path.join(path, 'maxrot0test_{}_{}.pkl'.format(config.test_size, config.patterns_reps))

    with tf.gfile.GFile(test_path, 'w') as f:
        pickle.dump(testset, f)


def generate_t2augtest(path, N=2):
    testset = []
    unifs = np.random.normal(size=N-1) / 0.5
    translations = np.zeros(N)
    translations[1:] = 10#unifs
    for i in range(N):
        gen_func = functools.partial(create_constellations,
                                shuffle_corners=config.shuffle_corners,
                                gaussian_noise=config.corner_noise,
                                max_rot=0,#config.max_rotation,
                                global_translation=translations[i],
                                max_upscale=config.pattern_upscale,
                                drop_prob=config.pattern_drop_prob,
                                which_patterns=patterns(config.patterns_reps),
                                rng=np.random.RandomState(
                                    seed=0)
                                )
        testset += create_dataset(gen_func, epoch_size=config.test_size, transform=lambda x: torch.tensor(x),
                         keys=['corners', 'presence', 'pattern_class_count'])
    test_path = os.path.join(path, 'maxrot0t2augtest_{}_{}_{}.pkl'.format(config.test_size, N, config.patterns_reps))
    print('test_path_aug', test_path)
    with tf.io.gfile.GFile(test_path, 'w') as f:
        pickle.dump(testset, f)

def generate_se2augtest(path, N=2):
    variables = ["gen_func%s" % i for i in range(0, N)]
    testset = []
    rng=np.random.RandomState(0)
    unifs= rng.normal(size=N-1) / 0.5
    translations = np.zeros(N)
    translations[1:] = 0 #unifs
    rotations = np.zeros(N)
    unifs_r = 10 / 180 * math.pi #rng.random(size=N-1) * 2 * math.pi
    print(unifs_r)
    rotations[1:] = unifs_r
    for i in range(N):
        gen_func = functools.partial(create_constellations,
                                shuffle_corners=config.shuffle_corners,
                                gaussian_noise=config.corner_noise,
                                max_rot=0,#config.max_rotation,
                                global_translation=translations[i],
                                global_rotation_angle=rotations[i],
                                max_upscale=config.pattern_upscale,
                                drop_prob=config.pattern_drop_prob,
                                which_patterns=patterns(config.patterns_reps),
                                rng=np.random.RandomState(
                                    seed=0)
                                )

        testset += create_dataset(gen_func, epoch_size=config.test_size, transform=lambda x: torch.tensor(x),
                         keys=['corners', 'presence', 'pattern_class_count'])
    test_path = os.path.join(path, 'testmaxrot0se2augtest_{}_{}_{}.pkl'.format(config.test_size, N, config.patterns_reps))
    with tf.io.gfile.GFile(test_path, 'w') as f:
        pickle.dump(testset, f)

if __name__ == '__main__':
    #generate_train('./data/constellation')
    #generate_test('./data/constellation')
    #generate_t2augtest('./data/constellation')
    generate_se2augtest('./data/constellation')


