from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.cellnet import *

learning_rate = 0.01
batchsize = 100
channels = 1
num_epoch = 1000

display_step = 1000
examples_to_show = 10
select_gpu = '2'


path = 'data/small'
configs = {
    "path": "data/small",
    "batchSize": batchsize,
    "imageSize": [128, 128],
    "channels": channels,
    "num_epoch": num_epoch,
    "display_step": display_step,
    "learning_rate": learning_rate,
    "select_gpu": select_gpu
}

model = CellNet(configs=configs)
model.train()












print(0)