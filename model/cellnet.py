import tensorflow as tf
import sys
import os
import cv2
import glob

# sys.path.append('utils')
from utils.layer import *

import matplotlib.pyplot as plt


class Data(object):

    def __init__(self, path=""):
        self.path = path
        self.train = ""
        self.test = ""
        self.valid = ""
        self.labels = {}
        self.prepare()

    def prepare(self):
        self.train_path = glob.glob(os.path.join(self.path, "train.txt"))[0]
        self.test_path = glob.glob(os.path.join(self.path, "test.txt"))[0]
        self.valid_path = glob.glob(os.path.join(self.path, "val.txt"))[0]
        labels_path = glob.glob(os.path.join(self.path, "labels.txt"))[0]
        labels = np.loadtxt(labels_path, delimiter=" ", dtype=str)
        for i in range(len(labels)):
            self.labels.update({labels[i, 0]: int(labels[i, 1])})

        self.train = self.read(self.train_path)
        self.test = self.read(self.test_path)
        self.valid = self.read(self.valid_path)

    def read(self, path):
        info = np.loadtxt(path, dtype=str, delimiter=" ")
        data = {}
        data.update({"path": info[:, 0]})
        data.update({"label": info[:, 1].astype(np.int)})
        return data


class CellNet(Data):

    def __init__(self, imgpath="", configs=None):
        Data.__init__(self, path=imgpath)
        # self.model = self.build(data)
        print(0)

    def build(self, x, is_training):
        with tf.variable_scope("cellnet"):
            x = conv2d(x, in_channels=1, out_channels=32, kernel_size=11, stride=2, padding="SAME")
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME")
            x = relu(x)
            x = batchnorm(x, is_training)

            x = conv2d(x, in_channels=32, out_channels=64, kernel_size=6, stride=2, padding="SAME")
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME")
            x = relu(x)
            x = batchnorm(x, is_training)

            x = conv2d(x, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding="SAME")
            x = maxpooling(x, poolsize=2, stride=2, padding="SAME")
            x = relu(x)
            x = batchnorm(x, is_training)

            x = flatten(x)
            x = fullyconnected(x, num_out=2048)
            x = relu(x)
            x = fullyconnected(x, num_out=64)

        return x

    def train(self):
        # self.build(x, is_training)
        return

    def test(self):
        return

    def evaluate(self):
        return