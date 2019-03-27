#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the classification model from "HEp-2 Cell Classification Based on A                  #
#       Deep Autoencoding-Classification Convolutional Neural Network" Jingxin Liu, Bolei Xu, Linlin Shen,             #
#       Jon Garibaldi, Guoping Qiu. IEEE 14th International Symposium on Biomedical Imaging (ISBI), 2017               #
#                                                                                                                      #
#   Citation:                                                                                                          #
#       HEp-2 Cell Classification Based on A Deep Autoencoding-Classification Convolutional Neural Network             #
#       Jingxin Liu, Bolei Xu, Linlin Shen, Jon Garibaldi, Guoping Qiu                                                 #
#       IEEE 14th International Symposium on Biomedical Imaging (ISBI), 2017                                           #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


# ==================================================================================================================== #
#                                             Model: Hep2Net (Liu et al.)                                              #
# ==================================================================================================================== #
from utils.layer import *


def liuhep2net(x, num_labels, is_training, verbose=False):

    '''
    The Hep2Net model from work "HEp-2 Cell Classification Based on A Deep Autoencoding-Classification
    Convolutional Neural Network" by Jingxin Liu, Bolei Xu, Linlin Shen, Jon Garibaldi, Guoping Qiu
    IEEE 14th International Symposium on Biomedical Imaging (ISBI), 2017

    :param x: tensor
            The input tensor with shape [batchsize, height, width, channel]
    :param num_labels: scatter
            A scatter indicates how many types of inputs.
    :param is_training: bool
    :param verbose: bool
    :return: tensor with shape [batchsize, num_labels]
    '''

    print("+---------------------------------------------------------------------------------------+")
    print("|                       Building the model -- Hep2Net (Liu et al.)                      |")
    print("+---------------------------------------------------------------------------------------+")
    print("|                                                                                       |")
    with tf.variable_scope("liuhep2net"):
        with tf.variable_scope("conv1"):
            x = conv2d(x, in_channels=1, out_channels=16, kernel_size=7, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv2"):
            x = conv2d(x, in_channels=16, out_channels=32, kernel_size=5, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv3"):
            x = conv2d(x, in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = conv2d(x, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
        with tf.variable_scope("conv4"):
            x = conv2d(x, in_channels=32, out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = conv2d(x, in_channels=64, out_channels=64, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            split = x

        #     x = batchnorm(x, is_training, verbose=verbose)
        #
        # with tf.variable_scope("fc7"):
        #     x = flatten(x)
        #     x = fullyconnected(x, num_out=150, verbose=verbose)
        #     x = 1.7159 * tanh((2.0 / 3.0) * x, verbose=verbose)
        #
        # with tf.variable_scope("out"):
        #     x = fullyconnected(x, num_out=num_labels, verbose=verbose)
        #     x = softmax(x, verbose=verbose)

    print("+---------------------------------------------------------------------------------------+")
    print("|                                  Model established                                    |")
    print("+---------------------------------------------------------------------------------------+")
    # net_variables = tf.trainable_variables(scope="liuhep2net")

    return x

