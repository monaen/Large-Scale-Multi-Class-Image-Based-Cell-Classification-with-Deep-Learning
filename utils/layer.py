#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the layers used in the "model.py" file                                               #
#                                                                                                                      #
#   Citation:                                                                                                          #
#       Large-scale Multi-class Image-based Cell Classification with Deep Learning                                     #
#       Nan Meng Edmund Y. Lam Kevin K. Tsia Hayden K.-H. So                                                           #
#       IEEE Journal of Biomedical and Health Informatics, 2018                                                        #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


# import systematic packages
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
        print("|--------------------------------------- Conv2D ----------------------------------------|")
        print("| feature size: {0: <28}  filter size: {1: <29}|".format(out.get_shape(), weight.get_shape()))

    return out

def upsampling2d(x, size=(2, 2), verbose=False):
    '''
    Upsampling layer

    :param x:
    :param size:
    :param verbose:
    :return:
    '''
    B, H, W, C = x.get_shape()
    out = tf.image.resize_nearest_neighbor(x, (size[0]*H, size[1]*W))

    if verbose:
        print("|------------------------------------ Upsampling2D -------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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

    if verbose:
        print("|---------------------------------------- ReLU -----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

    return out

def elu(x, verbose=False):
    '''
    ELU activation function

    :param x:
    :return:
    '''
    out = tf.nn.elu(x)

    if verbose:
        print("|---------------------------------------- ELU ------------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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

    if verbose:
        print("|---------------------------------------- LReLU ----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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

    if verbose:
        print("|-------------------------------------- LeakyReLU --------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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

    if verbose:
        print("|---------------------------------------- PRELU ----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

    return out

def sigmoid(x, verbose=False):
    '''
    Sigmoid activation function

    :param x:
    :return:
    '''
    out = tf.nn.sigmoid(x)

    if verbose:
        print("|--------------------------------------- Sigmoid ---------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

    return out

def tanh(x, verbose=False):
    '''
    Tanh activation function
    :param x:
    :param verbose:
    :return:
    '''
    out = tf.nn.tanh(x)

    if verbose:
        print("|----------------------------------------- Tanh ----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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

    if verbose:
        print("|------------------------------------- MaxPooling --------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

    return out


# ===================================================================== #
# ========================= Dropout Functions ========================= #
# ===================================================================== #

def dropout(x, keep_prob=0.5, verbose=False):
    '''
    Dropout Layer

    :param x:
    :param keep_prob:
    :param verbose:
    :return:
    '''
    out = tf.nn.dropout(x, keep_prob=keep_prob)

    if verbose:
        print("|-------------------------------------- Dropout ----------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

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
            out = tf.nn.batch_normalization(x, batch_mean, batch_var, beta, scale, epsilon)

            if verbose:
                print("|-------------------------------------- BatchNorm --------------------------------------|")
                print("| feature size: {0: <72}|".format(out.get_shape()))

            return out

    def bn_inference():
        out = tf.nn.batch_normalization(x, pop_mean, pop_var, beta, scale, epsilon)

        if verbose:
            print("|-------------------------------------- BatchNorm --------------------------------------|")
            print("| feature size: {0: <72}|".format(out.get_shape()))

        return out

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

def flatten(x):
    '''
    Flatten operation

    :param x:
    :return:
    '''
    input_shape = x.get_shape()
    dim = input_shape[1] * input_shape[2] * input_shape[3]
    transposed = tf.transpose(x, (0, 3, 1, 2))

    return tf.reshape(transposed, [-1, dim])


def fullyconnected(x, num_out, trainable=True, verbose=False):
    '''
    Fully-connected layer

    :param x: tensor with shape [batchsize, num_in]
    :param num_out: tensor with shape [batchsize, num_out]
    :param trainable: bool
    :param verbose: bool
    :return:
    '''
    num_in = x.get_shape()[-1]
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

    if verbose:
        print("|------------------------------------ Fullyconnect -------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))

    return out


# ===================================================================== #
# ========================= Softmax Functions ========================= #
# ===================================================================== #

def softmax(x, verbose=False):
    '''
    Softmax layer

    :param x:
    :param verbose:
    :return:
    '''
    out = tf.nn.softmax(x)

    if verbose:
        print("|--------------------------------------- Softmax ---------------------------------------|")
        print("| feature size: {0: <72}|".format(out.get_shape()))
    return out

