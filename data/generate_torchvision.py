# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
import h5py
import os
from torch.utils.data import DataLoader
import torchvision
import warnings

###############################################################################


def main():

    warnings.filterwarnings("ignore")

    # We parse the arguments
    arg_parser = ArgumentParser(description="Generate a torchvision dataset")
    arg_parser.add_argument(
        "dataset", metavar="dataset", type=str,
        help="Name of the dataset"
    )
    arg_parser.add_argument(
        "path", metavar="path", type=str,
        help="Path of the h5 dataset file"
    )
    arg_list = arg_parser.parse_args()
    dataset = arg_list.dataset
    dataset_path = arg_list.path

    # We load a folder as dataset
    if(os.path.exists(dataset)):
        data_train_list = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/train",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        data_test_list = torchvision.datasets.ImageFolder(
            root="./"+dataset+"/test",
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        test_size = len(data_test_list)
        train_size = len(data_train_list)

    # We load a torchvision dataset
    else:
        locals_ = locals()
        exec("dataset_fun = torchvision.datasets."+str(dataset),
             globals(), locals_)
        dataset_fun = locals_["dataset_fun"]

        data_train_list = dataset_fun(
            root="./data-"+dataset, train=True, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        data_test_list = dataset_fun(
            root="./data-"+dataset, train=False, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
            ])
        )
        test_size = data_test_list.data.shape[0]
        train_size = data_train_list.data.shape[0]

    # We get the train and test data
    data_train_loader = DataLoader(
        data_train_list,
        batch_size=train_size)
    data_test_loader = DataLoader(
        data_test_list, batch_size=test_size)
    data_train_list = list(data_train_loader)
    data_test_list = list(data_test_loader)
    input_train_list = data_train_list[0][0]
    label_train_list = data_train_list[0][1]
    input_test_list = data_test_list[0][0]
    label_test_list = data_test_list[0][1]

    # We create the dataset file
    dataset_file = h5py.File(dataset_path, "w")
    dataset_file["x_train"] = input_train_list
    dataset_file["y_train"] = label_train_list
    dataset_file["x_test"] = input_test_list
    dataset_file["y_test"] = label_test_list


if __name__ == "__main__":
    main()
