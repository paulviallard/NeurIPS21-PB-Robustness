#!/usr/bin/env python
# Author: Paul VIALLARD
#
# This file is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import glob

###############################################################################

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Nimbus Roman No9 L']
matplotlib.rcParams['font.style'] = "normal"
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['text.usetex'] = True

###############################################################################

attack_list = [
    "nothing", "uniformnoise", "uniformnoisepgd",
    "uniformnoiseiterativefgsm", "pgd", "iterativefgsm"]


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


bnd = []
tv = []

csv_list = glob.glob("result_bar/*.csv")
for i in range(len(csv_list)):
    csv_list[i] = pd.read_csv(csv_list[i], index_col=0)
    csv_list[i] = csv_list[i].to_dict()
    bnd.append(csv_list[i]["boundth7"])
    tv.append(csv_list[i]["tv_boundth7"])

for i in range(len(bnd)):
    name_list = bnd[i].keys()

    new_bnd = {}
    new_tv = {}

    for name in name_list:
        new_name = tuple(convert_name_into_pair(name))

        if(new_name[1] != "nothing"):
            new_bnd[new_name] = bnd[i][name]
            new_tv[new_name] = tv[i][name]

    bnd[i] = new_bnd
    tv[i] = new_tv

###############################################################################

sns.set(style="white", rc={"lines.linewidth": 3})
fig, ax_list = plt.subplots(
    1, 2, figsize=(20.0, 3.65), subplot_kw={'xticks': []})

latex_dict = {
    'uniformnoise': "{\sc unif}",
    'uniformnoisepgd': '{\sc pgd}$_{\sc U}$',
    'uniformnoiseiterativefgsm': '{\sc ifgsm}$_{\sc U}$',
    'nothing': '---'
}

bar_name_list = []
for name in bnd[0].keys():
    bar_name_list.append("("+latex_dict[name[0]]+", "+latex_dict[name[1]]+")")


# https://moonbooks.org/Articles/How-to-add-text-on-a-bar-with-matplotlib-/
def autolabel(rects):
    for idx, rect in enumerate(bar_plot):
        height = rect.get_height()
        if(height > 0.45):
            ax.text(rect.get_x() + rect.get_width()/2., 0.05*height,
                    bar_name_list[idx],
                    ha='center', va='bottom', rotation=90,
                    fontsize=21, color="white")
        else:
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                    bar_name_list[idx],
                    ha='center', va='bottom', rotation=90, fontsize=21)


ax = ax_list[0]
bar_plot = ax.bar(bar_name_list,
                  list(bnd[0].values()))
autolabel(bar_plot)
ax.bar(bar_name_list,
       list(tv[0].values()))
ax = ax_list[1]
bar_plot = ax.bar(bar_name_list,
                  list(bnd[1].values()))
autolabel(bar_plot)
ax.bar(bar_name_list,
       list(tv[1].values()))
plt.tight_layout()
fig.savefig("bar.pdf", bbox_inches="tight")
