import tensorflow as tf
import sys
import os
import cv2
import glob

# sys.path.append('utils')
from utils.layer import *
from utils.utils import *

import matplotlib.pyplot as plt
from tool.log_config import *
log_config()
tf.logging.set_verbosity(tf.logging.INFO)


class Data(object):

    def __init__(self, path=""):
        self.path = path
        self.traindata = None
        self.testdata = None
        self.validdata = None
        self.dic_labels = {}
        self.num_labels = None
        self.num_samples = None
        self.prepare()

    def prepare(self):
        self.train_path = glob.glob(os.path.join(self.path, "train.txt"))[0]
        self.test_path = glob.glob(os.path.join(self.path, "test.txt"))[0]
        self.valid_path = glob.glob(os.path.join(self.path, "val.txt"))[0]
        labels_path = glob.glob(os.path.join(self.path, "labels.txt"))[0]
        labels = np.loadtxt(labels_path, delimiter=" ", dtype=str)
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
        info = np.loadtxt(path, dtype=str, delimiter=" ")
        np.random.shuffle(info)
        data = {}
        data.update({"path": info[:, 0]})
        data.update({"label": info[:, 1].astype(np.int)})
        return data


class CellNet(Data):

    def __init__(self, configs=None):
        Data.__init__(self, path=configs["path"])
        self.is_training = tf.placeholder(tf.bool, [])
        self.lr = tf.placeholder(tf.float32, [])
        self.configs = configs
        self.inputs = tf.placeholder(tf.float32, [None, self.configs["imageSize"][0],
                                                  self.configs["imageSize"][1], self.configs["channels"]])
        self.labels = tf.placeholder(tf.float32, [None, self.num_labels])
        self.weights_folder = self.configs["weights_folder"]

        # ======== initialization ======== #
        os.environ["CUDA_VISIBLE_DEVICES"] = self.configs["select_gpu"]
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.num_epoch = self.configs["num_epoch"]

        # ======== predictions and loss ======== #
        self.predictions = self.build_model(self.inputs, self.is_training, verbose=True)
        self.loss = self.loss(predictions=self.predictions, labels=self.labels)

        self.saver = tf.train.Saver()


    def build_model(self, x, is_training, verbose=False):
        print("Building the model ......")
        with tf.variable_scope("cellnet"):
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
        self.net_variables = tf.trainable_variables(scope="cellnet")

        return x

    def loss(self, predictions, labels):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=predictions))
        return loss

    def loadbatch(self, data, iteration=0, state="train"):
        imgpaths, labels = data["path"], data["label"]
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

        batchimgs, batch_onehots = self.loading(batchimgpaths, batchlabels, state)

        return batchimgs, batch_onehots

    def loading(self, paths, labels, state):
        labels = reformat_label(labels, self.num_labels)
        b = self.configs["batchSize"]
        h = self.configs["imageSize"][0]
        w = self.configs["imageSize"][1]
        c = self.configs["channels"]
        batchimgs = np.zeros([b, h, w, c], dtype=np.float32)
        folder = self.configs["path"]
        for i in range(len(paths)):
            img = cv2.imread(os.path.join(folder, state, paths[i]), 0)
            img = cv2.resize(img, (w, h))
            img = img.astype(np.float32)
            img = img / img.max()
            if len(img.shape) == 2:
                img = np.expand_dims(img, -1)
            batchimgs[i] = img
        return batchimgs, labels

    def train(self, opt="adam", save_epoch=5, test_epoch=1, continues=False):
        if opt == "adam":
            opt = tf.train.AdamOptimizer(beta1=0.5, learning_rate=self.lr)

        train_op = opt.minimize(self.loss, global_step=self.global_step)
        lr = self.configs["learning_rate"]
        num_trainiter = self.num_trainsamples / self.configs["batchSize"]
        num_validiter = self.num_validsamples / self.configs["batchSize"]
        num_testiter = self.num_testsamples / self.configs["batchSize"]
        init = tf.global_variables_initializer()
        self.sess.run(init)

        if continues:
            self.saver.restore(self.sess, os.path.join(self.weights_folder, "cellnet/latest"))



        for epoch in range(self.num_epoch):

            # ============== Training ============== #
            for iteration in range(num_trainiter):
                batchimgs, batchlabels = self.loadbatch(self.traindata, iteration=iteration, state="train")

                _, loss, predictions = self.sess.run([train_op, self.loss, self.predictions],
                                                     feed_dict={self.inputs: batchimgs,
                                                                self.labels: batchlabels,
                                                                self.is_training: True,
                                                                self.lr: lr})
                acc = accuracy(predictions, batchlabels)
                logging.info("[Epoch {0:06d}][Iteration {1:08d}][Train]\t loss:{2:10.6f}\t accuracy: {3:.6f}".format(
                                                        epoch, iteration, loss, acc))

            # ============== Testing ============== #
            if epoch % test_epoch == 0:
                for iteration in range(num_testiter):
                    batchimgs, batchlabels = self.loadbatch(self.testdata, iteration=iteration, state="test")
                    loss, predictions = self.sess.run([self.loss, self.predictions],
                                                      feed_dict={self.inputs: batchimgs,
                                                                 self.labels: batchlabels,
                                                                 self.is_training: False})
                    acc = accuracy(predictions, batchlabels)
                    logging.info("[Epoch {0:06d}][Iteration {1:08d}][Test]\t loss:{2:10.6f}\t accuracy: {3:.6f}".format(
                                                            epoch, iteration, loss, acc))

            # ============== Save the model =============== #
            if (epoch !=0) and (epoch % save_epoch) == 0:
                fp = os.path.join(self.weights_folder, "cellnet")
                if not os.path.exists(fp):
                    os.makedirs(fp)
                save_path = self.saver.save(self.sess, os.path.join(self.weights_folder,
                                                                    "cellnet",
                                                                    "epoch{0:06d}".format(epoch) + ".ckpt"))
                self.saver.save(self.sess, os.path.join(self.weights_folder, "cellnet", "latest.ckpt"))
                logging.info("Model {} saved in file: {}".format("epoch{0:06d}".format(epoch) + ".ckpt", save_path))
        return

    # def test(self):
    #     return

    def evaluate(self):
        return