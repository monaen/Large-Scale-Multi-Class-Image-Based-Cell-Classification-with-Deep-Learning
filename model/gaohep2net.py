#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the classification model from "HEp-2 Cell Image Classification With                  #
#       Deep Convolutional Neural Networks" by Zhimin Gao, Lei Wang, Luping Zhou, Jianjia Zhang                        #
#       IEEE Journal of Biomedical and Health Informatics, 2017                                                        #
#                                                                                                                      #
#   Citation:                                                                                                          #
#       HEp-2 Cell Image Classification With Deep Convolutional Neural Networks                                        #
#       Zhimin Gao, Lei Wang, Luping Zhou, Jianjia Zhang                                                               #
#       IEEE Journal of Biomedical and Health Informatics, 2017                                                        #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


# ==================================================================================================================== #
#                                             Model: Hep2Net (Gao et al.)                                              #
# ==================================================================================================================== #
from utils.layer import *


def gaohep2net(x, num_labels, is_training, verbose=False):

    '''
    The Hep2Net model from work "HEp-2 Cell Image Classification With Deep Convolutional Neural Networks"
    by Zhimin Gao, Lei Wang, Luping Zhou, Jianjia Zhang
    IEEE Journal of Biomedical and Health Informatics, 2017

    :param x: tensor with shape [batchsize, height, width, channel]
    :param num_labels: tensor with shape [batchsize, num_labels]
    :param is_training: bool
    :param verbose: bool
    :return: tensor with shape [batchsize, num_labels]
    '''

    print("+---------------------------------------------------------------------------------------+")
    print("|                       Building the model -- Hep2Net (Gao et al.)                      |")
    print("+---------------------------------------------------------------------------------------+")
    print("|                                                                                       |")
    with tf.variable_scope("gaohep2net"):
        with tf.variable_scope("conv1"):
            x = conv2d(x, in_channels=1, out_channels=6, kernel_size=7, stride=1, padding="SAME", verbose=verbose)
            x = 1.7159 * tanh((2.0 / 3.0) * x, verbose=verbose)
        with tf.variable_scope("maxpool2"):
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("conv3"):
            x = conv2d(x, in_channels=6, out_channels=16, kernel_size=4, stride=2, padding="SAME", verbose=verbose)
            x = 1.7159 * tanh((2.0 / 3.0) * x, verbose=verbose)
        with tf.variable_scope("maxpool4"):
            x = maxpooling(x, poolsize=3, stride=3, padding="SAME", verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("conv5"):
            x = conv2d(x, in_channels=16, out_channels=32, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = 1.7159 * tanh((2.0 / 3.0) * x, verbose=verbose)
        with tf.variable_scope("maxpool6"):
            x = maxpooling(x, poolsize=3, stride=3, padding="SAME", verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("fc7"):
            x = flatten(x)
            x = fullyconnected(x, num_out=150, verbose=verbose)
            x = 1.7159 * tanh((2.0 / 3.0) * x, verbose=verbose)

        with tf.variable_scope("out"):
            x = fullyconnected(x, num_out=num_labels, verbose=verbose)
            x = softmax(x, verbose=verbose)

    print("+---------------------------------------------------------------------------------------+")
    print("|                                  Model established                                    |")
    print("+---------------------------------------------------------------------------------------+")
    # net_variables = tf.trainable_variables(scope="gaohep2net")

    return x

