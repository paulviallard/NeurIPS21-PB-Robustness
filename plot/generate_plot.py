#!/usr/bin/env python
# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


###############################################################################

attack_list = [
    "nothing", "uniformnoise", "uniformnoisepgd",
    "uniformnoiseiterativefgsm", "pgd", "iterativefgsm"
]


def convert_name_into_pair(name):
    name_list = name.split("_")
    pair_list = []
    for i in range(len(name_list)):
        if(name_list[i] in attack_list):
            pair_list.append(name_list[i])
    return pair_list


def convert_name_into_dataset(name):
    name_list = name.split("_")
    name = "_".join(name_list[0:3])
    return name

# --------------------------------------------------------------------------- #


csv = pd.read_csv("result_plot/merge_l2.csv", index_col=0)
csv_dict = csv.to_dict()

pair_list = [("uniformnoisepgd", "uniformnoisepgd"),
             ("uniformnoiseiterativefgsm", "uniformnoiseiterativefgsm"),
             ("uniformnoisepgd", "pgd"),
             ("uniformnoiseiterativefgsm", "iterativefgsm")]
color_list = sns.color_palette("viridis", 6)

new_csv_dict = {}
for metric in csv_dict.keys():
    for name, value in csv_dict[metric].items():
        name_1, name_2 = convert_name_into_pair(name)
        name_data = convert_name_into_dataset(name)
        if((name_1, name_2) not in pair_list):
            continue

        if(name_data not in new_csv_dict):
            new_csv_dict[name_data] = {}
        if((name_1, name_2) not in new_csv_dict[name_data]):
            new_csv_dict[name_data][(name_1, name_2)] = {}
        if(not(np.isnan(value))):
            new_csv_dict[name_data][(name_1, name_2)][metric] = value

data_list = sorted(list(new_csv_dict.keys()))
color_list = sns.color_palette("viridis", len(data_list))

csv_dict = new_csv_dict

marker_dict = {
    ("uniformnoisepgd", "uniformnoisepgd", "boundth6"): "o",
    ("uniformnoiseiterativefgsm", "uniformnoiseiterativefgsm",
     "boundth6"): "*",
    ("uniformnoisepgd", "uniformnoisepgd", "boundth7"): "^",
    ("uniformnoiseiterativefgsm", "uniformnoiseiterativefgsm",
     "boundth7"): "X"
}

###############################################################################

fig, ax_list = plt.subplots(1, 4, figsize=(10.0, 2.0))
fig.tight_layout(pad=0.4, w_pad=-2.2)

ax = ax_list[0]
ax.scatter(
    0.0, 0.0,
    color='none', s=50, edgecolor='none', facecolor='none')

for i in range(len(data_list)):
    name_data = data_list[i]

    for name in csv_dict[name_data].keys():

        name_bound = list(name)
        if(name[1] == "pgd" or name[1] == "iterativefgsm"):
            name_bound[1] = "uniformnoise"+name_bound[1]
            facecolors = color_list[i]
            name_bound = tuple(name_bound)
            name_marker = tuple(list(name_bound)+["boundth6"])

            ax.scatter(
                csv_dict[name_data][name_bound]["majorityvote"],
                csv_dict[name_data][name]["majorityvote"],
                color=color_list[i], s=50, edgecolor=color_list[i],
                marker=marker_dict[name_marker], facecolor=facecolors)

xlim_1 = np.array(ax.get_xlim())
xlim_1[0] = 0
ylim_1 = np.array(ax.get_ylim())
ylim_1[0] = 0
ax.plot([0, 1], [0, 1], linestyle='dashed', color='black')

ax.set_xlim(xlim_1)
ax.set_ylim(ylim_1)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

ax = ax_list[2]
for i in range(len(data_list)):
    name_data = data_list[i]
    for name in csv_dict[name_data].keys():

        name_bound = list(name)
        if(name[1] == "pgd" or name[1] == "iterativefgsm"):
            name_bound[1] = "uniformnoise"+name_bound[1]
            facecolors = color_list[i]
            name_bound = tuple(name_bound)
            name_marker = tuple(list(name_bound)+["boundth6"])

            ax.scatter(
                csv_dict[name_data][name_bound]["majorityvotemax"],
                csv_dict[name_data][name]["majorityvotemax"],
                color=color_list[i], s=50, edgecolor=color_list[i],
                marker=marker_dict[name_marker], facecolor=facecolors)
xlim_2 = np.array(ax.get_xlim())
xlim_2[0] = 0
ylim_2 = np.array(ax.get_ylim())
ylim_2[0] = 0

ax.plot([0, 1], [0, 1], linestyle='dashed', color='black')
ax.get_yaxis().set_visible(False)

ax.set_xlim(xlim_2)
ax.set_ylim(ylim_2)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)


###############################################################################

ax = ax_list[1]

ax.scatter(
    0.0, 0.0,
    color='none', s=50, edgecolor='none', facecolor='none')

for i in range(len(data_list)):
    name_data = data_list[i]
    for name in csv_dict[name_data].keys():

        name_bound = list(name)
        if(name[1] == "pgd" or name[1] == "iterativefgsm"):
            name_bound[1] = "uniformnoise"+name_bound[1]
            facecolors = color_list[i]
            name_bound = tuple(name_bound)
            name_marker = tuple(list(name_bound)+["boundth6"])

            ax.scatter(
                csv_dict[name_data][name_bound]["boundth6"],
                csv_dict[name_data][name]["majorityvote"],
                color=color_list[i], s=50, edgecolor=color_list[i],
                marker=marker_dict[name_marker], facecolor=facecolors)
ax.get_yaxis().set_visible(False)

xlim_1 = np.array(ax.get_xlim())
xlim_1[0] = 0
ylim_1 = np.array(ax.get_ylim())
ylim_1[0] = 0

ax.plot([0, 1], [0, 1], linestyle='dashed', color='black')

ax.set_xlim(xlim_1)
ax.set_ylim(ylim_1)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

ax = ax_list[3]

ax.scatter(
    0.0, 0.0,
    color='none', s=50, edgecolor='none', facecolor='none')

for i in range(len(data_list)):
    name_data = data_list[i]
    for name in csv_dict[name_data].keys():

        name_bound = list(name)
        if(name[1] == "pgd" or name[1] == "iterativefgsm"):
            name_bound[1] = "uniformnoise"+name_bound[1]
            facecolors = color_list[i]
            name_bound = tuple(name_bound)
            name_marker = tuple(list(name_bound)+["boundth6"])

            ax.scatter(
                csv_dict[name_data][name_bound]["boundth7"],
                csv_dict[name_data][name]["majorityvotemax"],
                color=color_list[i], s=50, edgecolor=color_list[i],
                marker=marker_dict[name_marker], facecolor=facecolors)
ax.get_yaxis().set_visible(False)

xlim_2 = np.array(ax.get_xlim())
xlim_2[0] = 0
ylim_2 = np.array(ax.get_ylim())
ylim_2[0] = 0
xlim = (0.0, 2.0)
ylim = (0.0, 2.0)

ax.plot([0, 1], [0, 1], linestyle='dashed', color='black')

ax.set_xlim(xlim_2)
ax.set_ylim(ylim_2)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=8)

for i in range(4):
    ax_list[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax_list[i].xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

###############################################################################

legend_list = []
for i in range(len(data_list)):
    legend_list.append(
        ax.scatter(
            -1, -1, marker='s',
            color=color_list[i]))

# --------------------------------------------------------------------------- #

fig.savefig("plot.pdf", bbox_inches="tight")
fig_legend = plt.figure(figsize=(8, 1))

c = len('Fashion COvsSH')
fig_legend.legend(
    legend_list,
    ["Fashion:COvsSH", "Fashion:SAvsBO", "Fashion:TOvsPU",
     "MNIST:1vs7", "MNIST:4vs9", "MNIST:5vs6"],
    frameon=False,  ncol=len(data_list), mode="expand")

fig_legend.savefig(
    'legend.pdf', format='pdf', bbox_inches="tight")
