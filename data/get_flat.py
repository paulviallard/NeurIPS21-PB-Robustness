# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import h5py
import numpy as np
import random
import warnings

###############################################################################


def main():

    # We initialize the seeds
    np.random.seed(42)
    random.seed(42)
    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(
        description="Flatten the input")
    arg_parser.add_argument(
        "old_path", metavar="old_path", type=str,
        help="Path of the h5 dataset file")
    arg_parser.add_argument(
        "new_path", metavar="new_path", type=str,
        help="Path of the new h5 dataset file")
    arg_list = arg_parser.parse_args()
    old_path = arg_list.old_path
    new_path = arg_list.new_path

    # We open the dataset (h5 file)
    dataset_file = h5py.File(old_path, "r")

    input_train = np.array(dataset_file["x_train"])
    label_train = np.array(dataset_file["y_train"])
    input_test = np.array(dataset_file["x_test"])
    label_test = np.array(dataset_file["y_test"])

    # We flatten the input
    input_train = np.reshape(input_train, (input_train.shape[0], -1))
    input_test = np.reshape(input_test, (input_test.shape[0], -1))

    # We save the new data
    if(old_path == new_path):
        dataset_file.close()
    dataset_file = h5py.File(new_path, "w")

    dataset_file["x_train"] = input_train
    dataset_file["y_train"] = label_train
    dataset_file["x_test"] = input_test
    dataset_file["y_test"] = label_test


if __name__ == "__main__":
    main()
