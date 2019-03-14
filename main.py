from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model.cellnet import *

learning_rate = 0.01
batchsize = 100
channels = 1
num_steps = 30000

display_step = 1000
examples_to_show = 10


path = 'data/small'
model = CellNet(path)













print(0)