#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the classification model from "Large-scale Multi-class Image-based Cell              #
#       Classification with Deep Learnin" by  Nan Meng, Edmund Y. Lam, Kevin K. Tsia, Hayden K.-H. So                  #
#       IEEE Journal of Biomedical and Health Informatics, 2018                                                        #
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


# ==================================================================================================================== #
#                                                   Model: CellNet                                                     #
# ==================================================================================================================== #
from utils.layer import *


def cellnet(x, num_labels, is_training, verbose=False):

    '''
    The CellNet model from work "Large-scale Multi-class Image-based Cell Classification with Deep Learning"
    by Nan Meng, Edmund Y. Lam, Kevin K. Tsia, Hayden K.-H. So
    IEEE Journal of Biomedical and Health Informatics, 2018

    :param x: tensor with shape [batchsize, height, width, channel]
    :param num_labels: tensor with shape [batchsize, num_labels]
    :param is_training: bool
    :param verbose: bool
    :return: tensor with shape [batchsize, num_labels]
    '''

    print("+---------------------------------------------------------------------------------------+")
    print("|                             Building the model -- CellNet                             |")
    print("+---------------------------------------------------------------------------------------+")
    print("|                                                                                       |")
    with tf.variable_scope("cellnet"):
        with tf.variable_scope("conv1"):
            x = conv2d(x, in_channels=1, out_channels=32, kernel_size=11, stride=2, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("conv2"):
            x = conv2d(x, in_channels=32, out_channels=64, kernel_size=6, stride=2, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("conv3"):
            x = conv2d(x, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="SAME", verbose=verbose)
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME", verbose=verbose)
            x = relu(x, verbose=verbose)
            x = batchnorm(x, is_training, verbose=verbose)

        with tf.variable_scope("fc1"):
            x = flatten(x)
            x = fullyconnected(x, num_out=2048, verbose=verbose)
            x = relu(x, verbose=verbose)

        with tf.variable_scope("fc2"):
            x = fullyconnected(x, num_out=64, verbose=verbose)
            x = relu(x, verbose=verbose)

        with tf.variable_scope("softmax"):
            x = fullyconnected(x, num_out=num_labels, verbose=verbose)
            x = softmax(x, verbose=verbose)
    print("+---------------------------------------------------------------------------------------+")
    print("|                                  Model established                                    |")
    print("+---------------------------------------------------------------------------------------+")
    # net_variables = tf.trainable_variables(scope="cellnet")

    return x

