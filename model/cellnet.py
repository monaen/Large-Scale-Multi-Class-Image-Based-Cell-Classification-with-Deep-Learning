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

    def __init__(self, imgpath=""):
        Data.__init__(self, path=imgpath)
        print(0)

    def build(self):
        return

    def train(self):
        return

    def test(self):
        return

    def evaluate(self):
        return