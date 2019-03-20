from __future__ import division, print_function, absolute_import
from model.classifier import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/Augmented", help="Path to the dataset")
parser.add_argument("--model_type", default="cellnet", help="Choose the model")
parser.add_argument("--channels", type=int, default=1, help="The number of input image channels")
parser.add_argument("--batchsize", type=int, default=3000, help="The number of input images in each batch")
parser.add_argument("--imgsize", type=int, default=128, help="The size of input images")
parser.add_argument("--learning_rate", type=float, default=0.00002, help="The learning rate for training")
parser.add_argument("--learning_rate_step", type=int, default=50)
parser.add_argument("--num_epoch", type=int, default=200)
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

    # ====== Start training ====== #
    model.train(num_epoch=args.num_epoch, save_epoch=args.save_step, continues=False)

    # ====== Start testing ====== #
    model.test()


if __name__ == "__main__":
    main(args)