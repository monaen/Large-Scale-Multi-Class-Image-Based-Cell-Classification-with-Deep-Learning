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

def conv2d(x, in_channels, out_channels, kernel_size=3, stride=1, padding="SAME", trainable=True, verbose=False):
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

def maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=False):
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
                         padding=padding)
    return out


# ===================================================================== #
# ========================= BatchNorm Functions ======================= #
# ===================================================================== #

def batchnorm(x, is_training, decay=0.99, epsilon=0.001, trainable=True, verbose=False):
    '''
    Batch Normalization layer

    :param x:
    :param is_training:
    :param decay:
    :param epsilon:
    :param trainable:
    :return:
    '''
    def bn_train():
        batch_mean, batch_var = tf.nn.moments(x, axes=[0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

    def bn_inference():
        return tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

    dim = x.get_shape().as_list()[-1]
    beta = tf.get_variable(
        name='beta',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.0),
        trainable=trainable)
    scale = tf.get_variable(
        name='scale',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=0.1),
        trainable=trainable)
    pop_mean = tf.get_variable(
        name='pop_mean',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(0.0),
        trainable=False)
    pop_var = tf.get_variable(
        name='pop_var',
        shape=[dim],
        dtype=tf.float32,
        initializer=tf.constant_initializer(1.0),
        trainable=False)
    return tf.cond(is_training, bn_train, bn_inference)


# ===================================================================== #
# ===================== Fully-connected Functions ===================== #
# ===================================================================== #

def flatten(x, verbose=False):
    '''
    Flatten operation

    :param x:
    :param verbose:
    :return:
    '''
    input_shape = tf.shape(x)
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))

    return tf.reshape(transposed, [-1, dim])


def fullyconnected(x, num_out, trainable=True, verbose=False):
    '''
    Fully-connected layer

    :param x:
    :return:
    '''
    num_in = tf.shape(x)[-1]
    W = tf.get_variable(name="weight",
                        shape=[num_in, num_out],
                        dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1),
                        trainable=trainable)
    b = tf.get_variable(name="bias",
                        shape=[num_out],
                        dtype=tf.float32,
                        initializer=tf.constant_initializer(0.0),
                        trainable=trainable)
    out = tf.add(tf.matmul(x, W), b)

    return out





