from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.models import *

learning_rate = 0.0002
batchsize = 3000
channels = 1
num_epoch = 100

display_step = 1
select_gpu = '3'
results_folder = "results"
weights_folder = "weights"

path = 'data/Augmented'
configs = {
    "path": path,
    "batchSize": batchsize,
    "imageSize": [128, 128],
    "channels": channels,
    "display_step": display_step,
    "learning_rate": learning_rate,
    "select_gpu": select_gpu,
    "results_folder": results_folder,
    "weights_folder": weights_folder
}

model = CellNet(configs=configs)
model.train(num_epoch=num_epoch)
model.test()











print(0)