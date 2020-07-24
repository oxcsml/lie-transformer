'''Generate constellation data and write data to a file'''
import sys
sys.path.append('forge')
sys.path.append('.')
import os
import pickle
import forge
from forge import flags
import ipdb
from configs.constellation import create_constellations
import tensorflow as tf
config = forge.config()

def generate_examples(path):
    trainset = create_constellations(size_n=config.train_size, shuffle_corners=config.shuffle_corners, gaussian_noise=config.corner_noise,
        max_rot=config.max_rotation_train,
        max_upscale=config.pattern_upscale,
        drop_prob=config.pattern_drop_prob,
        which_patterns=config.patterns_train.split(','),
        )

    testset = create_constellations(size_n=config.test_size, shuffle_corners=config.shuffle_corners,
        gaussian_noise=config.corner_noise,
        max_rot=config.max_rotation_test,
        max_upscale=config.pattern_upscale,
        drop_prob=config.pattern_drop_prob,
        which_patterns=config.patterns_test.split(','),
        )


    train_data = trainset['corners'], trainset['presence'], trainset['pattern_class_count']
    test_data = testset['corners'], testset['presence'], testset['pattern_class_count']

    train_path = os.path.join(path, 'train.pkl')
    test_path = os.path.join(path, 'test.pkl')

    with tf.gfile.GFile(train_path, 'w') as f:
       pickle.dump(train_data, f)

    with tf.gfile.GFile(test_path, 'w') as f:
        pickle.dump(test_data, f)

if __name__ == '__main__':
    generate_examples('/Users/charlinelelan/eqv_transformer/data/constellation')


