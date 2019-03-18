from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.models import *

learning_rate = 0.0002
batchsize = 1000
channels = 1
num_epoch = 100

display_step = 10
select_gpu = '3'
results_folder = "results"
weights_folder = "weights"

path = 'data/augmented'
configs = {
    "path": "data/small",
    "batchSize": batchsize,
    "imageSize": [128, 128],
    "channels": channels,
    "num_epoch": num_epoch,
    "display_step": display_step,
    "learning_rate": learning_rate,
    "select_gpu": select_gpu,
    "results_folder": results_folder,
    "weights_folder": weights_folder
}

model = CellNet(configs=configs)
model.train()












print(0)