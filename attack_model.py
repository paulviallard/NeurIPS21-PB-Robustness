#!/usr/bin/env python
# Author: Guillaume VIDOT (AIRBUS SAS)
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import re
import argparse
import numpy as np
import logging
from learner.gradient_descent_learner import GradientDescentLearner

from core.metrics import Metrics
from core.save import save_csv
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
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)

    arg_parser = argparse.ArgumentParser(description='')

    # Add argument parser
    arg_parser.add_argument(
        "data", metavar="data", type=str,
        help="data")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "model", metavar="model",
        type=str, help="model")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "save", metavar="save", type=str,
        help="save")
    arg_parser.add_argument(
        "name", metavar="name", type=str,
        help="name"
    )
    arg_parser.add_argument(
        "metricName", metavar="metricName", type=str,
        help="metricName"
    )
    # ----------------------------------------------------------------------- #
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
        "--load", metavar="load", default=None, type=str,
        help="load")
    arg_parser.add_argument(
        "--batch-size", metavar="batch-size", default=64, type=int,
        help="batch-size")
    # ----------------------------------------------------------------------- #
    arg_parser.add_argument(
        "--attack", metavar="attack", default=None, type=str,
        help="attack")
    # ----------------------------------------------------------------------- #

    # Retrieve argument
    arg_list = arg_parser.parse_args()
    data = arg_list.data

    save = arg_list.save
    name_exp = arg_list.name
    metric_name = arg_list.metricName

    model = arg_list.model
    opt_model = arg_list.opt_model

    metric = arg_list.metric
    opt_metric = arg_list.opt_metric

    batch_size = arg_list.batch_size
    name_attack = arg_list.attack
    # ----------------------------------------------------------------------- #

    # Load the attacked dataset
    dataset = File("data/attack_signed/"+data+".h5", "r")

    x_attack = np.array(dataset["x_attack"])
    y_attack = np.array(dataset["y_attack"])

    device = "cuda"
    new_device = torch.device('cpu')
    if(torch.cuda.is_available() and device != "cpu"):
        new_device = torch.device(device)

    model_kwargs = get_dict_arg(opt_model)
    model = Module(model, new_device, **model_kwargs)
    model.to(new_device)

    attacks = ['nothing', 'pgd', 'iterativefgsm']
    if name_attack not in attacks:
        # Reshape only when we sample noises using PGDU, IFGSMU or UNIF
        nb_data = x_attack.shape[0]
        nb_noise = x_attack.shape[1]
        x_attack = x_attack.reshape(-1, x_attack.shape[2])
        y_attack = y_attack.reshape(-1, nb_noise)
        y_attack = y_attack[:, 0:1]

    metric_kwargs = get_dict_arg(opt_metric)
    metric_kwargs.update({"name": metric, "model": model})
    metric = Metrics(**metric_kwargs)

    learner = GradientDescentLearner(
        model, None, None, None, new_device,
        batch_size=batch_size, epoch=0)

    if(arg_list.load is not None):
        load_dict = pickle.load(open("data/model_signed/"+arg_list.load, "rb"))
        model.load_state_dict(load_dict["model_param"])
        if(load_dict["arg_list"].metric == metric):
            metric.load(load_dict["metric_param"])

    # Handle the output function regarding the version of the bound of Th 7
    if(metric_name == "boundth7" or metric_name == "boundth7tv"):
        x_attack = learner.output(x_attack, out_type="list")
    else:
        x_attack = learner.output(x_attack)

    # Reshape the attacked input regarding the attack used.
    # (either the classical ones or ours)
    if name_attack not in attacks:
        x_attack = x_attack.reshape(nb_data, nb_noise, -1)
    else:
        x_attack = np.expand_dims(x_attack, axis=1)

    # Save the results in the right format regarding the metric used.
    detail_metric = ["boundth6", "boundth7", "boundth7tv"]
    if metric_name in detail_metric:
        save_csv(save, {
            metric_name: metric.fit(x_attack, y_attack).item(),
            "r_"+metric_name: metric.r.item(),
            "div_"+metric_name: metric.div.item(),
            "tv_"+metric_name: metric.tv.item()},
            name_exp, erase=True)
    else:
        save_csv(save, {
            metric_name: metric.fit(x_attack, y_attack).item()},
            name_exp, erase=True)

###############################################################################
