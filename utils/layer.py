import tensorflow as tf
import numpy as np


def initializer_conv2d(in_channels, out_channels, mapsize, stddev_factor=1.0, mode="Glorot"):
    '''
    Initialization in the style of Glorot 2010.

    :param in_channels:
    :param out_channels:
    :param mapsize:
    :param stddev_factor: should be 1.0 for linear activations, and 2.0 for ReLUs [default: 1.0]
    :param mode: [default: "Glorot"]
    :return:
    '''
    if mode == "Glorot":
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels*out_channels)*mapsize*mapsize))
    else:
        stddev = 1.0

    return tf.truncated_normal([mapsize, mapsize, in_channels, out_channels],
                               mean=0.0, stddev=stddev)

def Conv2d(x, in_channel, out_channel, kernel_size=3, stride=1, padding="SAME", trainable=True):
    '''
    Convolutional layer

    :param x:
    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param stride:
    :param padding:
    :param trainable:
    :return:
    '''

