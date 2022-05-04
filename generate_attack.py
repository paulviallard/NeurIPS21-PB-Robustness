#!/usr/bin/env python
# Author: Guillaume VIDOT (AIRBUS SAS)
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import numpy as np
import logging
from learner.attack_gradient_descent_learner import AttackGradientDescentLearner

from core.metrics import Metrics
from core.attack import Attack
from models.models import Module

from h5py import File
import torch
import pickle
import random


###############################################################################

def get_dict_arg(arg):

    if(arg is not None):
        arg = re.split('=|,', arg)
    else:
        arg = []
    arg_str = "dict_arg = {"
    for i in range(0, len(arg), 2):
        arg_str += "\""+arg[i]+"\": "
        arg_str += arg[i+1]+","
    arg_str += "}"
    locals = {}
    exec(arg_str, globals(), locals)
    dict_arg = locals["dict_arg"]

    return dict_arg


###############################################################################


if __name__ == "__main__":

    ###########################################################################
    logging.basicConfig(level=logging.INFO)
    #  logging.getLogger().disabled = True
    logging.StreamHandler.terminator = ""

    SEED = 0

    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    arg_parser = argparse.ArgumentParser(description='')

    # Add argument parser
    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "mode", metavar="mode", type=str,
        help="mode")
    arg_parser.add_argument(
        "target", metavar="target", type=str,
        help="target")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--start", metavar="start", type=int, default=None,
        help="start")
    arg_parser.add_argument(
        "--end", metavar="end", type=int, default=None,
        help="end")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "model", metavar="model",
        type=str, help="model")
    arg_parser.add_argument(
        "--opt-model", metavar="opt-model", default=None,
        type=str, help="opt-model")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--attack", metavar="attack", default="nothing",
        type=str, help="attack")
    arg_parser.add_argument(
        "--opt-attack", metavar="opt-attack", default=None,
        type=str, help="opt-attack")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--load", metavar="load", default=None, type=str,
        help="load")
    # ----------------------------------------------------------------------- #

    # Retrieve argument
    arg_list = arg_parser.parse_args()
    data = arg_list.data
    mode = arg_list.mode
    target = arg_list.target

    start = arg_list.start
    end = arg_list.end

    model = arg_list.model
    opt_model = arg_list.opt_model

    attack = arg_list.attack
    opt_attack = arg_list.opt_attack
    if(opt_attack == ""):
        opt_attack = None

    # ----------------------------------------------------------------------- #

    # Load the dataset
    dataset = File("data/"+data+".h5", "r")

    x_attack = np.array(dataset["x_"+mode])
    y_attack = np.array(dataset["y_"+mode])

    if(start is None and end is not None):
        x_attack = x_attack[:end, :]
        y_attack = y_attack[:end]
    if(start is not None and end is None):
        x_attack = x_attack[start:, :]
        y_attack = y_attack[start:]
    if(start is not None and end is not None):
        x_attack = x_attack[start:end, :]
        y_attack = y_attack[start:end]

    y_attack = np.expand_dims(y_attack, 1)

    device = "cuda"
    new_device = torch.device('cpu')
    if(torch.cuda.is_available() and device != "cpu"):
        new_device = torch.device(device)

    device = new_device
    model_kwargs = get_dict_arg(opt_model)
    model = Module(model, device, **model_kwargs)
    model.to(device)

    metric = Metrics("attackgibbs", model)

    if(arg_list.load is not None):
        load_dict = pickle.load(open("data/model_signed/"+arg_list.load, "rb"))
        model.load_state_dict(load_dict["model_param"])
        if(load_dict["arg_list"].metric == metric):
            metric.load(load_dict["metric_param"])

    # Initialize the attack
    attack_kwargs = get_dict_arg(opt_attack)
    attack = Attack(attack, model, device, metric.fit, **attack_kwargs)
    x_attack, y_attack = attack.fit(x_attack, y_attack)

    dataset.close()

    # Store the attacked dataset
    dataset = File("data/attack_signed/"+target+".h5", "w")

    # Reshape the attacked input regarding the attack used.
    # (either the classical ones or ours)
    name_attack = ['nothing', 'pgd', 'iterativefgsm']
    if attack.name not in name_attack:
        x_attack = x_attack.reshape(-1, attack.nb_noise, x_attack.shape[1])

    dataset["x_attack"] = x_attack
    dataset["y_attack"] = y_attack

    if hasattr(attack, "x_orig") and hasattr(attack, "adversarial"):
        # when attack == nothing no need of it
        dataset["x_orig"] = attack.x_orig
        dataset["adversarial"] = attack.adversarial

###############################################################################
