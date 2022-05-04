#!/usr/bin/env python
# Author: Guillaume VIDOT
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import numpy as np
import logging
import pickle
from learner.attack_gradient_descent_learner import AttackGradientDescentLearner
from learner.early_stopping_learner import EarlyStoppingLearner

from sklearn.metrics import zero_one_loss

from core.metrics import Metrics
from core.attack import Attack
from models.models import Module

from h5py import File
import torch
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
    torch.cuda.manual_seed_all(SEED)
    torch.manual_seed(SEED)

    arg_parser = argparse.ArgumentParser(description='')

    # Add argument parser
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    arg_parser.add_argument(
        "--train-start", metavar="train-start", default=None, type=int,
        help="train-start")
    arg_parser.add_argument(
        "--train-end", metavar="train-end", default=None, type=int,
        help="train-start")
    arg_parser.add_argument(
        "--val-start", metavar="val-start", default=None, type=int,
        help="val-start")
    arg_parser.add_argument(
        "--val-end", metavar="val-end", default=None, type=int,
        help="valid-end")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "model", metavar="model",
        type=str, help="model")
    arg_parser.add_argument(
        "--opt-model", metavar="opt-model", default=None,
        type=str, help="opt-model")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--metric", metavar="metric", default="gibbs",
        type=str, help="metric")
    arg_parser.add_argument(
        "--opt-metric", metavar="opt-metric", default=None,
        type=str, help="opt-metric")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--optimizer", metavar="optimizer", default="adam",
        type=str, help="optimizer")
    arg_parser.add_argument(
        "--opt-optimizer", metavar="opt-optimizer", default=None,
        type=str, help="opt-optimizer")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--attack", metavar="attack", default="nothing",
        type=str, help="attack")
    arg_parser.add_argument(
        "--opt-attack", metavar="opt-attack", default=None,
        type=str, help="opt-attack")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--epoch", metavar="epoch", default=10, type=int,
        help="epoch")
    arg_parser.add_argument(
        "--val-epoch", metavar="val-epoch", default=0, type=int,
        help="valid")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--lr", metavar="lr", default=0.00005, type=float,
        help="lr")
    arg_parser.add_argument(
        "--batch-size", metavar="batch-size", default=64, type=int,
        help="batch-size")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--load", metavar="load", default=None, type=str,
        help="load")
    arg_parser.add_argument(
        "--save", metavar="save", default=None, type=str,
        help="save")
    # ----------------------------------------------------------------------- #

    # Retrieve argument
    arg_list = arg_parser.parse_args()

    train_start = arg_list.train_start
    train_end = arg_list.train_end
    val_start = arg_list.val_start
    val_end = arg_list.val_end

    model = arg_list.model
    opt_model = arg_list.opt_model

    metric = arg_list.metric
    opt_metric = arg_list.opt_metric

    optimizer_name = arg_list.optimizer
    opt_optimizer = arg_list.opt_optimizer

    attack = arg_list.attack
    opt_attack = arg_list.opt_attack
    if(opt_attack == ""):
        opt_attack = None

    epoch = arg_list.epoch
    val_epoch = arg_list.val_epoch

    lr = arg_list.lr
    batch_size = arg_list.batch_size

    # ----------------------------------------------------------------------- #

    # Load the dataset
    data = File("data/"+arg_list.data+".h5", "r")

    x_train = np.array(data["x_train"])
    y_train = np.array(data["y_train"])
    x_val = None
    y_val = None

    # Split the data into training and validation set
    if(val_start is None and val_end is not None):
        x_val = x_train[:val_end, :]
        y_val = y_train[:val_end]
    elif(val_start is not None and val_end is not None):
        x_val = x_train[val_start:val_end, :]
        y_val = y_train[val_start:val_end]
    elif(val_start is not None and val_end is None):
        x_val = x_train[val_start:, :]
        y_val = y_train[val_start:]

    if(train_start is None and train_end is not None):
        x_train = x_train[:train_end, :]
        y_train = y_train[:train_end]
    elif(train_start is not None and train_end is not None):
        x_train = x_train[train_start:train_end, :]
        y_train = y_train[train_start:train_end]
    elif(train_start is not None and train_end is None):
        x_train = x_train[train_start:, :]
        y_train = y_train[train_start:]

    device = "cuda"
    new_device = torch.device('cpu')
    if(torch.cuda.is_available() and device != "cpu"):
        new_device = torch.device(device)
    device = new_device

    model_kwargs = get_dict_arg(opt_model)
    model = Module(model, device, **model_kwargs)
    model.to(new_device)

    metric_kwargs = get_dict_arg(opt_metric)
    metric_kwargs.update({"name": metric, "model": model})
    metric = Metrics(**metric_kwargs)

    optimizer_kwargs = get_dict_arg(opt_optimizer)

    if optimizer_name == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)

    if(metric.param is not None):
        param_list = list(model.parameters())
        param_list.append(metric.param)
        param_list = torch.nn.ParameterList(param_list)

        if optimizer_name == "adam":
            optim = torch.optim.Adam(param_list, lr=lr)

    # Initialize the attack
    metric_attack = Metrics("attackgibbs", model)
    attack_kwargs = get_dict_arg(opt_attack)
    attack = Attack(attack, model, device, metric_attack.fit, **attack_kwargs)

    # Initialize the model
    criteria = Metrics("attackgibbs", model)
    learner = AttackGradientDescentLearner(
        model, metric.fit, zero_one_loss, attack, optim, device,
        batch_size=batch_size, epoch=epoch)
    if(val_epoch > 0 and x_val is not None and y_val is not None):
        learner = EarlyStoppingLearner(
            learner, criteria.fit, val_epoch=val_epoch)

    if(val_epoch > 0 and x_val is None and y_val is None):
        learner = EarlyStoppingLearner(
            learner, metric.fit, val_epoch=val_epoch)

    if(arg_list.load is not None):
        load_dict = pickle.load(open("data/model/"+arg_list.load, "rb"))
        learner.load(load_dict["model_param"], beginning=True)
        if(load_dict["arg_list"].metric == metric):
            metric.load(load_dict["metric_param"])

    if(val_epoch > 0 and x_val is not None and y_val is not None):
        learner = learner.fit(x_train, y_train, x_val, y_val)
    else:
        learner = learner.fit(x_train, y_train, x_train, y_train)

    # Save the model parameters
    if(arg_list.save is not None):
        pickle.dump({
            "model_param": learner.save(),
            "metric_param": metric.save(),
            "arg_list": arg_list
        }, open("data/model/"+arg_list.save, "wb"))
        pickle.dump(learner.list_loss, open("data/loss/"+arg_list.save, "wb"))

###############################################################################
