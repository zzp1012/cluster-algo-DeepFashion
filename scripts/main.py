import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
# cwd change to current file's dir
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np

# import internal libs
from data import DataUtils
from model import ModelUtils
from evaluation import Evaluator
from utils import get_datetime, set_logger, get_logger, log_settings

def add_args() -> argparse.Namespace:
    """get arguments from the program.

    Returns:
        return a dict containing all the program arguments 
    """
    parser = argparse.ArgumentParser(
        description="simple verification")
    ## the basic setting of exp
    parser.add_argument("--save_root", default="../outs/tmp/", type=str,
                        help='the path of saving results.')
    parser.add_argument("--seed", default=0, type=int,
                        help="set the seed.")
    parser.add_argument("--model", default="kmeans", type=str, choices=["kmeans", "spectral"],
                        help="the clustering model.")
    parser.add_argument("--n_clusters", default=50, type=int,
                        help="the number of clusters.")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="enable debug info output.")
    args = parser.parse_args()

    if not os.path.exists(args.save_root):
        os.makedirs(args.save_root)

    # set the save_path
    exp_name = "-".join([get_datetime(),
                         f"seed{args.seed}",
                         f"{args.model}",
                         f"n_clusters{args.n_clusters}"])
    args.save_path = os.path.join(args.save_root, exp_name)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    return args


def main():
    # get the args.
    args = add_args()
    # set the logger
    set_logger(args.save_path)
    # get the logger
    logger = get_logger(__name__, verbose=args.verbose)
    # show the args.
    log_settings(args)

    # load the data
    features, labels = DataUtils.read()

    # load the model(clustering algorithm)
    model = ModelUtils(args.model, n_clusters=args.n_clusters, verbose=args.verbose)

    logger.info("#######fit and predict the data.")
    # test the time
    pred_labels = model.predict(features)
    # save the results
    np.save(os.path.join(args.save_path, "pred_labels.npy"), pred_labels)

    # evaluate the model
    logger.info("#######evaluating the model.")
    for metric in ["pairwise", "bcubed"]:
        evaluator = Evaluator(metric)
        avg_pre, avg_rec, avg_f1 = evaluator(labels, pred_labels)
        logger.info(f"""{metric} score
            avg_pre: {avg_pre}, avg_rec: {avg_rec}, avg_f1: {avg_f1}""")


if __name__ == "__main__":
    main()