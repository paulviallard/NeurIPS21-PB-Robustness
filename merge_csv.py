#!/usr/bin/env python
# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import glob
import os
import argparse
import numpy as np
import logging
import torch
from core.save import save_csv
import random
import pandas as pd

###############################################################################


if __name__ == "__main__":

    ###########################################################################
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    SEED = 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

    arg_parser = argparse.ArgumentParser(description='')

    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "folder", metavar="folder", type=str,
        help="folder")
    arg_parser.add_argument(
        "save", metavar="save", type=str,
        help="save")

    # ----------------------------------------------------------------------- #

    arg_list = arg_parser.parse_args()
    folder = arg_list.folder
    save = arg_list.save

    csv_list = glob.glob(os.path.abspath(folder)+"/*.csv")

    for csv_path in csv_list:
        csv_name = os.path.basename(csv_path)
        csv_name = csv_name[:-4]
        metric = csv_name.split("_")[-1]
        exp_name = "_".join(csv_name.split("_")[:-1])
        data = pd.read_csv(csv_path, index_col=0)
        columns = np.array(data.columns)
        values = data.to_numpy()[0]

        for (key, value) in zip(columns, values):
            save_csv(save, {key: value}, exp_name, erase=True)

###############################################################################
