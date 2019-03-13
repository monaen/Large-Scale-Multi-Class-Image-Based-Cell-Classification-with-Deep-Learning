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
        stddev = np.sqrt(stddev_factor / (np.sqrt(in_channels * out_channels) * mapsize * mapsize))
    else:
        stddev = 1.0

    return tf.truncated_normal([mapsize, mapsize, in_channels, out_channels],
                               mean=0.0, stddev=stddev)

def Conv2d(x, in_channels, out_channels, kernel_size=3, stride=1, padding="SAME", trainable=True, verbose=False):
    '''
    Convolutional layer

    :param x:
    :param in_channel:
    :param out_channel:
    :param kernel_size:
    :param stride:
    :param padding:
    :param trainable:
    :return: Tensor
    '''
    weight = initializer_conv2d(in_channels, out_channels, kernel_size)
    filter = tf.get_variable(name="weight",
                             initializer=weight,
                             dtype=tf.float32,
                             trainable=trainable)

    out = tf.nn.conv2d(input=x,
                       filter=filter,
                       strides=[1, stride, stride, 1],
                       padding=padding)

    if verbose:
        print("|----------------------- Conv2D ----------------------")
        print("|--- filter size: [] ------ feature size: []")
        print(weight.get_shape())
        print(out.get_shape())

    return out


# ===================================================================== #
# ======================= Activation Functions ======================== #
# ===================================================================== #

def relu(x, verbose=False):
    '''
    ReLU activation function

    :param x:
    :return:
    '''
    out = tf.nn.relu(x)

    return out


def elu(x, verbose=False):
    '''
    ELU activation function

    :param x:
    :return:
    '''
    out = tf.nn.elu(x)

    return out

def lrelu(x, trainable=None, verbose=False):
    '''
    LReLU activation function

    :param x:
    :param trainable:
    :return:
    '''
    alpha = 0.2
    out = tf.maximum(alpha * x, x)

    return out

def leakyrelu(x, leak=0.2, verbose=False):
    '''
    LeakyReLU activation function

    :param x:
    :param leak:
    :return:
    '''
    t1 = 0.5 * (1 + leak)
    t2 = 0.5 * (1 - leak)
    out = t1 * x + t2 * tf.abs(x)

    return out


def prelu(x, trainable=True, verbose=False):
    '''
    PReLU activation function

    :param x:
    :param trainable:
    :return:
    '''
    alpha = tf.get_variable(name="alpha", shape=x.get_shape()[-1], dtype=tf.float32,
                            initializer=tf.constant_initializer(0.0),
                            trainable=trainable)
    out = tf.maximum(0.0, x) + alpha * tf.minimum(0.0, x)

    return out

def sigmoid(x, verbose=False):
    '''
    Sigmoid activation function

    :param x:
    :return:
    '''
    out = tf.nn.sigmoid(x)

    return out

# ===================================================================== #
# ========================= Pooling Functions ========================= #
# ===================================================================== #

def maxpooling(x, poolsize=2, stride=2, verbose=False):
    '''
    Max Pooling Layer

    :param x:
    :param poolsize:
    :param stride:
    :param verbose:
    :return:
    '''
    out = tf.nn.max_pool(value=x,
                         ksize=[1, poolsize, poolsize, 1],
                         strides=[1, stride, stride, 1],
                         padding="SAME")
    return out

def batchnorm(x, is_training, decay=0.99, epsilon=0.001, trainable=True):
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + )