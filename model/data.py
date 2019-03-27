#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the class "Data" which is used to store the data information                         #
#                                                                                                                      #
#   Contact:                                                                                                           #
#       Nan Meng                                                                                                       #
#       naen.mong@gmail.com                                                                                            #
#       University of Hong Kong                                                                                        #
# -------------------------------------------------------------------------------------------------------------------- #


# ==================================================================================================================== #
#                                                         Data                                                         #
# ==================================================================================================================== #
import numpy as np
import glob
import os
from numpy import genfromtxt

class Data(object):

    def __init__(self, path=""):
        self.path = path
        self.traindata = None
        self.testdata = None
        self.validdata = None
        self.dic_labels = {}
        self.num_labels = 4
        self.num_samples = None
        if self.path != "":
            self.prepare()

    def prepare(self):
        self.train_path = glob.glob(os.path.join(self.path, "Train.txt"))[0]
        self.test_path = glob.glob(os.path.join(self.path, "Test.txt"))[0]
        self.valid_path = glob.glob(os.path.join(self.path, "Valid.txt"))[0]
        labels_path = glob.glob(os.path.join(self.path, "labels.txt"))[0]
        labels = genfromtxt(labels_path, dtype=str)
        for i in range(len(labels)):
            self.dic_labels.update({labels[i, 0]: int(labels[i, 1])})

        self.num_labels = len(labels)
        self.traindata = self.read(self.train_path)
        self.num_trainsamples = len(self.traindata["path"])
        self.testdata = self.read(self.test_path)
        self.num_testsamples = len(self.testdata["path"])
        self.validdata = self.read(self.valid_path)
        self.num_validsamples = len(self.validdata["path"])

    def read(self, path):
        info = genfromtxt(path, delimiter=" ", dtype=str)
        np.random.shuffle(info)
        data = {}
        data.update({"path": info[:, 0]})
        data.update({"label": info[:, 1].astype(np.int)})
        return data