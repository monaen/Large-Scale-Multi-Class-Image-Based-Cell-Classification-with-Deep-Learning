#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the classification model from "Convolutional Neural Network for Cell                 #
#       Classification Using Microscope Images of Intracellular Actin Networks" Ronald Wihal Oei, Guanqun Hou,         #
#       Fuhai Liu, Jin Zhong, Jiewen Zhang, Zhaoyi An, Luping Xu, Yujiu Yang. PloS one, March 13, 2019                 #
#                                                                                                                      #
#   Citation:                                                                                                          #
#       Convolutional Neural Network for Cell Classification Using Microscope Images of Intracellular Actin Networks   #
#       Ronald Wihal Oei, Guanqun Hou, Fuhai Liu, Jin Zhong, Jiewen Zhang, Zhaoyi An, Luping Xu, Yujiu Yang            #
#       PloS one, March 13, 2019                                                                                       #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


# ==================================================================================================================== #
#                                             Model: Hep2Net (Oei et al.)                                              #
# ==================================================================================================================== #
from utils.layer import *


def oeinet(x, num_labels, is_training, verbose=False):

    '''
    The model from work "Convolutional Neural Network for Cell Classification Using Microscope Images of
    Intracellular Actin Networks" by Ronald Wihal Oei, Guanqun Hou, Fuhai Liu, Jin Zhong, Jiewen Zhang,
    Zhaoyi An, Luping Xu, Yujiu Yang. PloS one, March 13, 2019

    :param x: tensor
            The input tensor with shape [batchsize, height, width, channel]
    :param num_labels: scatter
            A scatter indicates how many types of inputs.
    :param is_training: bool
    :param verbose: bool
    :return: tensor with shape [batchsize, num_labels]
    '''

    print("+---------------------------------------------------------------------------------------+")
    print("|                       Building the model -- OeiNet (Oei et al.)                       |")
    print("+---------------------------------------------------------------------------------------+")
    print("|                                                                                       |")
    with tf.variable_scope("oeinet"):
        with tf.variable_scope("conv1_1"):
            x = conv2d(x, in_channels=1,  out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv1_2"):
            x = conv2d(x, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv2_1"):
            x = conv2d(x, in_channels=64,  out_channels=128, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv2_2"):
            x = conv2d(x, in_channels=128, out_channels=128, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv3_1"):
            x = conv2d(x, in_channels=128, out_channels=256, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv3_2"):
            x = conv2d(x, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv3_3"):
            x = conv2d(x, in_channels=256, out_channels=256, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv4_1"):
            x = conv2d(x, in_channels=256, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv4_2"):
            x = conv2d(x, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv4_3"):
            x = conv2d(x, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv5_1"):
            x = conv2d(x, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv5_2"):
            x = conv2d(x, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv5_3"):
            x = conv2d(x, in_channels=512, out_channels=512, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("fc1"):
            x = flatten(x)
            x = fullyconnected(x, num_out=512, verbose=verbose)
            x = dropout(x, keep_prob=0.8)
        with tf.variable_scope("out"):
            x = fullyconnected(x, num_out=num_labels, verbose=verbose)
            x = dropout(x, keep_prob=0.8)
            x = softmax(x, verbose=verbose)

    print("+---------------------------------------------------------------------------------------+")
    print("|                                  Model established                                    |")
    print("+---------------------------------------------------------------------------------------+")
    # net_variables = tf.trainable_variables(scope="oeinet")

    return x

