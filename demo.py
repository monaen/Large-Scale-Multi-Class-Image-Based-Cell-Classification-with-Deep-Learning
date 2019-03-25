#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script runs a demo for multi-class cell classification of CellNet.                                 #
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

from __future__ import division, print_function, absolute_import
from model.classifier import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="", help="Path to the dataset")
parser.add_argument("--model_type", default="cellnet", help="Choose the model")
parser.add_argument("--channels", type=int, default=1, help="The number of input image channels")
parser.add_argument("--batchsize", type=int, default=3000, help="The number of input images in each batch")
parser.add_argument("--imgsize", type=int, default=128, help="The size of input images")
parser.add_argument("--learning_rate", type=float, default=0.00002, help="The learning rate for training")
parser.add_argument("--learning_rate_step", type=int, default=50)
parser.add_argument("--num_epoch", type=int, default=100)
parser.add_argument("--display_step", type=int, default=1)
parser.add_argument("--save_step", type=int, default=5)
parser.add_argument("--select_gpu", default='3')
parser.add_argument("--results_folder", default="results")
parser.add_argument("--weights_folder", default="weights")

args = parser.parse_args()


def main(args):

    # ====== Load configurations ====== #
    configs = {
        "path": args.dataset,
        "batchSize": args.batchsize,
        "imageSize": [args.imgsize, args.imgsize],
        "channels": args.channels,
        "display_step": args.display_step,
        "learning_rate": args.learning_rate,
        "lr_step": args.learning_rate_step,
        "select_gpu": args.select_gpu,
        "results_folder": args.results_folder,
        "weights_folder": args.weights_folder
    }

    # ====== Model definition ====== #
    model = Classifier(args.model_type, configs=configs)

    # ====== Evaluation on validation set ====== #
    # model.valid()

    # ====== Evaluation on test set ====== #
    # model.test()

    # ====== Evaluation on samples ====== #
    labeldic = {"THP1": 0, "MCF7": 1, "MB231": 2, "PBMC": 3}
    imglist = glob.glob('samples/*.jpg')
    imglist.sort()
    imgs, labels = readsamples(imglist=imglist, labeldic=labeldic)

    predictions = model.evaluate(img=imgs)
    print("Classification Accuracy: {} %".format(np.sum(predictions == labels)/len(predictions)*100))
    for i in range(len(imglist)):
        print("Cell IMG: [{}] \t Prediction: {}".format(imglist[i], labeldic.keys()[(labeldic.values().index(predictions[i]))]))


def readsamples(imglist, labeldic):
    labels = []
    batchimgs = np.zeros([len(imglist), 128, 128, 1], dtype=np.float32)
    for i in range(len(imglist)):
        name = imglist[i].split("/")[-1].split('_')[0]
        labels.append(labeldic[name])
        tmpimg = cv2.imread(imglist[i], 0)
        tmpimg = tmpimg.astype(np.float32)
        tmpimg = (tmpimg - np.min(tmpimg)) / (np.max(tmpimg) - np.min(tmpimg))
        batchimgs[i, :, :, 0] = tmpimg

    labels = np.array(labels)
    return batchimgs, labels



if __name__ == "__main__":
    main(args)