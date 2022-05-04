# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Note: The source code of this file is based on [1]
from re import sub
from torch.utils.data import Dataset
import torch
import numpy as np
import copy


class NumpyDataset(Dataset):

    def __init__(self, x_y_dict):
        """
        Initialize the dataset

        Parameters
        ----------
        dataset_dict: dict
            The dictionary of datasets
        """
        self._dataset_key = list(x_y_dict.keys())

        self._mode_list = []

        # We remove the modes in self._dataset_key
        new_dataset_key = {}
        for key in self._dataset_key:
            new_key = sub("_[^_]+$", "", key)
            if(new_key != key):
                new_dataset_key[new_key] = None
        self._dataset_key = list(new_dataset_key.keys())

        # and we create a copy of the dataset
        self._dataset_dict = copy.deepcopy(x_y_dict)

        # We initialize the mode by default
        self._mode = "train"

    def set_mode(self, mode):
        """
        Change the "mode"

        Parameters
        ----------
        mode: str
            The "mode" of the dataset (train or test)
        """
        # We set the mode of the dataset
        self._mode = mode

    def get_mode(self):
        """
        Get the "mode"

        Return
        ------
        str
        The "mode" of the dataset (train or test)
        """
        # We get the mode of the dataset
        return self._mode

    def get_mode_dataset(self, key):
        """
        Get the input/label (the "key") of the according "mode"

        Parameters
        ----------
        key: str
            The name that represents either the input or the label

        Return
        ------
        ndarray
        The numpy array corresponding to the key in the dataset
        """
        # We get the name of the key for the dictionary
        if(self._mode == "train" or self._mode == "test"):
            mode_dict_key = key+"_"+self._mode
        else:
            raise RuntimeError("mode must be either train or test")

        # We get the numpy array in the dictionary
        if(mode_dict_key in self._dataset_dict):
            return self._dataset_dict[mode_dict_key]

        return self._dataset_dict[mode_dict_key]

    def __len__(self):
        """
        Get the size of a dataset (of a given "mode")

        Return
        ------
        int
        The size of the dataset (of a given mode)
        """
        return len(self.get_mode_dataset("x"))

    def class_size(self):
        """
        Get the number of labels

        Return
        ------
        int
        The number of labels
        """
        if("y"+self._mode in self._dataset_dict):
            return len(np.unique(self.get_mode_dataset("y")))
        return 1

    def __getitem__(self, i):
        """
        Get the i-th example
        Parameters
        ----------
        i: int
            The index of the example

        Return
        -------
        dict
        A dictionary with the input or the labels
        """

        # We get the input or the labels
        item_dict = {
            "mode": self._mode,
            "size": self.__len__(),
            "class_size": self.class_size()
        }
        for key in self._dataset_key:
            item_dict[key] = torch.tensor(self.get_mode_dataset(key)[i])

        return item_dict

###############################################################################

# References:
#  [1] https://github.com/paulviallard/ECML21-PB-CBound/ (under MIT license)
