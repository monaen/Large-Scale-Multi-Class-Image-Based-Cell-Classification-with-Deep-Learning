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
import re
import cv2
import matplotlib.pyplot as plt
from utils.layer import *
from utils.utils import *
from model.data import *
from tool.log_config import *


class CellNet(Data):

    def __init__(self, configs=None):
        Data.__init__(self, path=configs["path"])
        self.model_type = "cellnet"
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.configs = configs
        self.inputs = tf.placeholder(tf.float32, [None, self.configs["imageSize"][0],
                                                  self.configs["imageSize"][1], self.configs["channels"]])
        self.labels = tf.placeholder(tf.float32, [None, self.num_labels])
        self.weights_folder = self.configs["weights_folder"]
        self.display_step = self.configs["display_step"]
        self.lr_step = self.configs["lr_step"]

        # ======== initialization ======== #
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs["select_gpu"]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        # ======== predictions and loss ======== #
        self.predictions = self.build_model(self.inputs, self.is_training, verbose=True)
        self.loss = self.loss(predictions=self.predictions, labels=self.labels)

        self.saver = tf.train.Saver()


    def build_model(self, x, is_training, verbose=False):
        print("Building the model ......")
        with tf.variable_scope(self.model_type):
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

            with tf.variable_scope("fc3"):
                x = fullyconnected(x, num_out=self.num_labels, verbose=verbose)
                x = relu(x, verbose=verbose)
        print("Model established.")
        self.net_variables = tf.trainable_variables(scope=self.model_type)

        return x

    def loss(self, predictions, labels):
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
                logging.info("[Epoch {0:06d}][Iteration {1:08d}][Train]\t learning rate: {2:10.9f}\t loss:{3:10.6f}\t "
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
                    logging.info("[Epoch {0:06d}][Iteration {1:08d}][Valid]\t loss:{2:10.6f}\t accuracy: "
                                 "{3:.6f}".format(epoch, iteration, loss, acc))

            # ============== Save the model =============== #
            fp = os.path.join(self.weights_folder, self.model_type)
            if not os.path.exists(fp):
                os.makedirs(fp)
            self.saver.save(self.sess, os.path.join(self.weights_folder, self.model_type, "latest.ckpt"))
            logging.info("Model {} saved in file: {}".format("epoch{0:06d}".format(epoch) + ".ckpt", save_path))
            self.save_state(epoch, lr, acc, loss)

            if (epoch != 0) and (epoch % save_epoch) == 0:
                save_path = self.saver.save(self.sess, os.path.join(self.weights_folder,
                                                                    self.model_type,
                                                                    "epoch{0:06d}".format(epoch) + ".ckpt"))

            if (epoch != 0) and (epoch % self.display_step) == 0:
                trainacc, trainloss, validacc, validloss = self.read_current_acc_loss_log()
                self.acc_loss_plot(trainacc, trainloss, title="Train")
                self.acc_loss_plot(validacc, validloss, title="Validate")
        return

    def test(self):
        num_testiter = self.num_testsamples / self.configs["batchSize"]
        for iteration in range(num_testiter):
            batchimgs, batchlabels = self.loadbatch(self.testdata, iteration=iteration)
            loss, predictions = self.sess.run([self.loss, self.predictions],
                                              feed_dict={self.inputs: batchimgs,
                                                         self.labels: batchlabels,
                                                         self.is_training: False})
            acc = accuracy(predictions, batchlabels)
            logging.info("Testing ----- [Iteration {0:08d}][Test]\t loss:{1:10.6f}\t accuracy: {2:.6f}".format(iteration, loss, acc))
        return

    def evaluate(self, img, model_path):
        self.saver.restore(self.sess, os.path.join(self.weights_folder, os.path.join(self.model_type, model_path)))
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=0)
            img = np.expand_dims(img, exis=-1)
        prediction = self.sess.run([self.predictions], feed_dict={self.inputs: img, self.is_training:False})
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
        expression_train = "Train.*loss:  ([0-9]*.[0-9]*).*accuracy: ([0-9]*.[0-9]*)"
        expression_valid = "Valid.*loss:  ([0-9]*.[0-9]*).*accuracy: ([0-9]*.[0-9]*)"
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
        plt.savefig(os.path.join(self.configs["results_folder"], title+'__AccurateLoss.pdf'), interpolation='nearest',
                    transparent=True, bbox_inches='tight')
        return