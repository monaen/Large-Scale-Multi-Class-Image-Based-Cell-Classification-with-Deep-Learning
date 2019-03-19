from __future__ import division, print_function, absolute_import
from model.models import *
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/Augmented")
parser.add_argument("--channels", type=int, default=1)
parser.add_argument("--batchsize", type=int, default=3000)
parser.add_argument("--imgsize", type=int, default=128)
parser.add_argument("--learning_rate", type=float, default=0.0002)
parser.add_argument("--learning_rate_step", type=int, default=10)
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
    model = CellNet(configs=configs)

    # ====== Start training ====== #
    model.train(num_epoch=args.num_epoch, save_epoch=args.save_step)

    # ====== Start testing ====== #
    model.test()




if __name__ == "__main__":
    main(args)






print(0)