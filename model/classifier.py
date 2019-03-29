#!/usr/bin/env python
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------------------------------------------- #
#   Description:                                                                                                       #
#       This python script define the classification models according to the corresponding work.                       #
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


# import systematic packages
import cv2
import re

# import utils packages (self-made)
from utils.utils import *

# import different models
from model.data import *
from model.cellnet import *
from model.gaohep2net import *
from model.liuhep2net import *
from model.oeinet import *

# import packages for debugging
import matplotlib.pyplot as plt

# import logging packages
from tool.log_config import *

# ====== logging setup ====== #
log_config()
tf.logging.set_verbosity(tf.logging.INFO)


########################################################################################################################
#                                                                                                                      #
#                                                  Classes Definition                                                  #
#                                                                                                                      #
########################################################################################################################

class Classifier(Data):

    def __init__(self, model_type="", configs=None, verbose=True):
        Data.__init__(self, path=configs["path"])
        self.model_type = model_type.lower()
        self.configs = configs

        # ============ get settings ============ #
        self.weights_folder = self.configs["weights_folder"]
        self.display_step = self.configs["display_step"]
        self.lr_step = self.configs["lr_step"]

        # =========== initialization =========== #
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.inputs = tf.placeholder(tf.float32, [None, self.configs["imageSize"][0], self.configs["imageSize"][1],
                                                  self.configs["channels"]])
        self.labels = tf.placeholder(tf.float32, [None, self.num_labels])

        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs["select_gpu"]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # ======== predictions and loss ======== #
        self.predictions = self.build_model(verbose=verbose)
        self.loss = self.getloss(predictions=self.predictions, labels=self.labels)
        self.saver = tf.train.Saver()

    def build_model(self, verbose):
        '''
        Construct the learning model for cell classification

        :param verbose: bool
                    control whether print the model framework in Terminal.

        :return: predictions: tensor
                    The prediction tensor with shape [batchsize, num_label] indicate the probability of each class.
        '''
        if self.model_type == "cellnet":
            predictions = cellnet(self.inputs, self.num_labels, self.is_training, verbose)

        elif self.model_type == "gaohep2net":
            predictions = gaohep2net(self.inputs, self.num_labels, self.is_training, verbose)

        elif self.model_type == "liuhep2net":
            self.recons, predictions = liuhep2net(self.inputs, self.num_labels, self.is_training, verbose)

        elif self.model_type == "oeinet":
            predictions = oeinet(self.inputs, self.num_labels, self.is_training, verbose)

        else:
            predictions = None

        return predictions

    def getloss(self, predictions, labels):
        if self.model_type == "liuhep2net":
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions)) + \
                   tf.nn.l2_loss(self.recons-self.inputs)
        else:
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions))
        return loss

    def loadbatch(self, data, iteration=0):
        imgpaths, labels = data["path"], data["label"]
        if np.random.random() > 0.9:
            idx = np.random.permutation(len(imgpaths))
            imgpaths, labels = imgpaths[idx], labels[idx]
        batchsize = self.configs["batchSize"]
        num_files = len(imgpaths)

        start = iteration * batchsize % num_files
        end = (iteration+1) * batchsize % num_files
        if start >= end:
            batchimgpaths = imgpaths[start:] + imgpaths[:end]
            batchlabels = labels[start:] + labels[:end]
        else:
            batchimgpaths = imgpaths[start:end]
            batchlabels = labels[start:end]

        batchimgs, batch_onehots = self.loading(batchimgpaths, batchlabels)

        return batchimgs, batch_onehots

    def loadtestbatch(self, data, iteration=0):
        imgpaths, labels = data["path"], data["label"]
        batchsize = self.configs["batchSize"]

        start = iteration * batchsize
        end = (iteration+1) * batchsize
        batchimgpaths = imgpaths[start:end]
        batchlabels = labels[start:end]

        batchimgs, batch_onehots = self.loading(batchimgpaths, batchlabels)

        return batchimgs, batch_onehots

    def loading(self, paths, labels):
        labels = reformat_label(labels, self.num_labels)
        b = self.configs["batchSize"]
        h = self.configs["imageSize"][0]
        w = self.configs["imageSize"][1]
        c = self.configs["channels"]
        batchimgs = np.zeros([b, h, w, c], dtype=np.float32)
        folder = self.configs["path"]
        for i in range(len(paths)):
            img = cv2.imread(os.path.join(folder, paths[i]), 0)
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32)
            img = img / img.max()
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
            batchimgs[i] = img
        return batchimgs, labels

    def adjust_learning_rate(self, lr, epoch, step=5):
        return lr * (0.1 ** (epoch // step))

    def train(self, num_epoch=10, opt="adam", save_epoch=5, test_epoch=1, continues=False, model_path="latest.ckpt"):
        if opt == "adam":
            opt = tf.train.AdamOptimizer(beta1=0.5, learning_rate=self.lr)

        train_op = opt.minimize(self.loss, global_step=self.global_step)
        lr_start = self.configs["learning_rate"]
        num_trainiter = self.num_trainsamples / self.configs["batchSize"]
        num_validiter = self.num_validsamples / self.configs["batchSize"]

        init = tf.global_variables_initializer()
        self.sess.run(init)

        epoch_start = 0
        if continues:
            self.saver.restore(self.sess, os.path.join(self.weights_folder, os.path.join(self.model_type, model_path)))
            _, epoch_start = self.read_state()
            epoch_start += 1

        for epoch in range(epoch_start, num_epoch):
            # -------------------------------------- #
            #                Training                #
            # -------------------------------------- #
            for iteration in range(num_trainiter):
                # ================ read batch images and labels ================ #
                batchimgs, batchlabels = self.loadbatch(self.traindata, iteration=iteration)
                # ================ adjust the learning rate ================ #
                lr = self.adjust_learning_rate(lr_start, epoch, step=self.lr_step)

                # ================ running the model ================ #
                _, loss, predictions = self.sess.run([train_op, self.loss, self.predictions],
                                                     feed_dict={self.inputs: batchimgs,
                                                                self.labels: batchlabels,
                                                                self.is_training: True,
                                                                self.lr: lr})
                acc = accuracy(predictions, batchlabels)
                logging.info("[Epoch {0:06d}][Iteration {1:08d}][Train]\t learning rate: {2:10.9f}\t loss: {3:10.6f}\t "
                             "accuracy: {4:.6f}".format(epoch, iteration, lr, loss, acc))

            # -------------------------------------- #
            #                 Testing                #
            # -------------------------------------- #
            if epoch % test_epoch == 0:
                for iteration in range(num_validiter):
                    batchimgs, batchlabels = self.loadbatch(self.validdata, iteration=iteration)
                    loss, predictions = self.sess.run([self.loss, self.predictions],
                                                      feed_dict={self.inputs: batchimgs,
                                                                 self.labels: batchlabels,
                                                                 self.is_training: False})
                    acc = accuracy(predictions, batchlabels)
                    logging.info("[Epoch {0:06d}][Iteration {1:08d}][Valid]\t loss: {2:10.6f}\t accuracy: "
                                 "{3:.6f}".format(epoch, iteration, loss, acc))

            # ============== Save the model =============== #
            fp = os.path.join(self.weights_folder, self.model_type)
            if not os.path.exists(fp):
                os.makedirs(fp)
            save_path = self.saver.save(self.sess, os.path.join(self.weights_folder, self.model_type, "latest.ckpt"))
            logging.info("Model {} saved in file: {}".format("epoch{0:06d}".format(epoch) + ".ckpt", save_path))
            self.save_state(epoch, lr, acc, loss)

            if (epoch != 0) and (epoch % save_epoch) == 0:
                self.saver.save(self.sess, os.path.join(self.weights_folder, self.model_type,
                                                        "epoch{0:06d}".format(epoch) + ".ckpt"))
                logging.info("Model {} saved in file: {}".format("epoch{0:06d}".format(epoch) + ".ckpt", save_path))

            if (epoch != 0) and (epoch % self.display_step) == 0:
                trainacc, trainloss, validacc, validloss = self.read_current_acc_loss_log()
                self.acc_loss_plot(trainacc, trainloss, title="Train")
                self.acc_loss_plot(validacc, validloss, title="Validate")
        return

    def test(self, modelpath=""):
        if modelpath != "":
            self.saver.restore(self.sess, modelpath)
        else:
            self.saver.restore(self.sess, os.path.join(self.weights_folder, self.model_type, "latest.ckpt"))
        batchsize =  self.configs["batchSize"]
        num_testiter = self.num_testsamples / batchsize
        Predictions = np.array([], dtype=np.float32)
        TestLabels = np.array([], dtype=np.float32)
        for iteration in range(num_testiter):
            batchimgs, batchlabels = self.loadtestbatch(self.testdata, iteration=iteration)
            loss, predictions = self.sess.run([self.loss, self.predictions],
                                              feed_dict={self.inputs: batchimgs,
                                                         self.labels: batchlabels,
                                                         self.is_training: False})
            acc = accuracy(predictions, batchlabels)
            logging.info("Testing ----- [Iteration {0:08d}][Test]\t loss: {1:10.6f}\t "
                         "accuracy: {2:.6f}".format(iteration, loss, acc))
            if iteration == 0:
                Predictions = predictions
                TestLabels = batchlabels
            else:
                Predictions = np.concatenate([Predictions, predictions], axis=0)
                TestLabels = np.concatenate([TestLabels, batchlabels], axis=0)
        Accuracy = accuracy(Predictions, TestLabels)
        logging.info("The overall accuracy for entire Test dataset is: {0:8.6f} %".format(Accuracy))
        return

    def valid(self, modelpath=""):
        if modelpath != "":
            self.saver.restore(self.sess, modelpath)
        else:
            self.saver.restore(self.sess, os.path.join(self.weights_folder, self.model_type, "latest.ckpt"))
        batchsize = self.configs["batchSize"]
        num_validiter = self.num_validsamples / batchsize
        Predictions = np.array([], dtype=np.float32)
        TestLabels = np.array([], dtype=np.float32)
        for iteration in range(num_validiter):
            batchimgs, batchlabels = self.loadtestbatch(self.validdata, iteration=iteration)
            loss, predictions = self.sess.run([self.loss, self.predictions],
                                              feed_dict={self.inputs: batchimgs,
                                                         self.labels: batchlabels,
                                                         self.is_training: False})
            acc = accuracy(predictions, batchlabels)
            logging.info("Validation ----- [Iteration {0:08d}][Valid]\t loss: {1:10.6f}\t "
                         "accuracy: {2:.6f}".format(iteration, loss, acc))
            if iteration == 0:
                Predictions = predictions
                TestLabels = batchlabels
            else:
                Predictions = np.concatenate([Predictions, predictions], axis=0)
                TestLabels = np.concatenate([TestLabels, batchlabels], axis=0)
        Accuracy = accuracy(Predictions, TestLabels)
        logging.info("The overall accuracy for entire Validation dataset is: {0:8.6f} %".format(Accuracy))
        return


    def evaluate(self, img, model_path='latest.ckpt'):
        self.saver.restore(self.sess, os.path.join(self.weights_folder, os.path.join(self.model_type, model_path)))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, axis=-1)
        prediction = self.sess.run(self.predictions, feed_dict={self.inputs: img, self.is_training: False})
        prediction = np.argmax(prediction, 1)
        return prediction


    def read_state(self):
        statefile = open(os.path.join(self.weights_folder, self.model_type+".state"), 'r')
        lines = statefile.read()
        lines = lines.split("\n")
        model_type = lines[0].split(":")[-1]
        epoch = int(lines[1].split(":")[-1])
        statefile.close()
        return model_type, epoch

    def save_state(self, epoch, lr, acc, loss):
        statefile = open(os.path.join(self.weights_folder, self.model_type+".state"), "w")
        statefile.write("model:{}\n".format(self.model_type))
        statefile.write("latest epoch:{}\n".format(str(epoch)))
        statefile.write("learning rate:{}\n".format(str(lr)))
        statefile.write("latest mini-batch accuracy:{}\n".format(str(acc)))
        statefile.write("latest mini-batch loss:{}\n".format(str(loss)))
        statefile.close()
        return

    def read_current_acc_loss_log(self):
        logfile = open("./log/current--logging.log", "r")
        lines = logfile.readlines()
        expression_train = "Train.*loss: ([0-9]*.[0-9]*).*accuracy: ([0-9]*.[0-9]*)"
        expression_valid = "Valid.*loss: ([0-9]*.[0-9]*).*accuracy: ([0-9]*.[0-9]*)"
        TrainAcc = []
        ValidAcc = []
        TrainLoss = []
        ValidLoss = []

        for i in range(len(lines)):
            train_message = re.search(expression_train, lines[i])
            if train_message is not None:
                trainloss, trainacc = train_message.group(1), train_message.group(2)
                TrainLoss.append(trainloss)
                TrainAcc.append(trainacc)
            valid_message = re.search(expression_valid, lines[i])
            if valid_message is not None:
                validloss, validacc = valid_message.group(1), valid_message.group(2)
                ValidLoss.append(validloss)
                ValidAcc.append(validacc)
        logfile.close()

        TrainLoss = np.array(TrainLoss).astype(np.float32)
        TrainAcc = np.array(TrainAcc).astype(np.float32)
        ValidLoss = np.array(ValidLoss).astype(np.float32)
        ValidAcc = np.array(ValidAcc).astype(np.float32)
        return TrainAcc, TrainLoss, ValidAcc, ValidLoss

    def acc_loss_plot(self, acc, loss, title="Example"):
        fig, ax1 = plt.subplots()
        ax1.plot(np.arange(len(acc)), acc, "r--", label="Accuracy")
        ax1.set_xlabel('iteration')
        ax1.set_ylabel('Accuracy', color="red")
        ax1.tick_params(axis='y', labelcolor="red")

        ax2 = ax1.twinx()
        ax2.plot(np.arange(len(loss)), loss, "b-",  label="Loss")
        ax2.set_ylabel('Loss', color="blue")
        ax2.tick_params(axis='y', labelcolor="blue")
        fig.tight_layout()
        fig.legend(loc=1, bbox_to_anchor=(1, 0.28), bbox_transform=ax1.transAxes)
        plt.title(title)
        if not os.path.exists(self.configs["results_folder"]):
            os.makedirs(self.configs["results_folder"])
        plt.savefig(os.path.join(self.configs["results_folder"], title+'_Accuracy&Loss.pdf'), interpolation='nearest',
                    transparent=True, bbox_inches='tight')
        plt.close("all")
