#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# In[2]:
def format_number_with_unicode_spaces(number):
    number_str = str(number)
    formatted_str = ""

    for i, digit in enumerate(number_str):
        formatted_str += digit

        # Insert a zero-width space after every 3 digits, except for the last group
        if (len(number_str) - i - 1) % 3 == 0 and i != len(number_str) - 1:
            formatted_str += "\u2009"  # Unicode zero-width space

    return formatted_str


data = {
    "Dist-MPNet-ParaCrawl": {
        "x": ["0", "1000", "5000", "15000", "50000", "100000", "500000", "None"],
        "y": [42.33, 42.830000000000005, 42.815, 43.61750000000001, 44.17, 44.29, 45.2025, 45.60000000000001],
        "std": [
            0,
            0.3188259713385966,
            0.6013526419664256,
            0.11627015954233376,
            0.24062418831031934,
            0.077781745930518,
            0.23541187310753894,
            0.036055512754640105,
        ],
    },
    "TSDAE-Small-E-Czech": {
        "x": ["0", "1000", "5000", "15000", "50000", "100000", "500000", "None"],
        "y": [40.54, 41.7, 42.575, 42.985, 43.5275, 43.8575, 44.5925, 44.98],
        "std": [
            0,
            0.12786711852544505,
            0.16007810593582206,
            0.12757350822172916,
            0.05539629951540129,
            0.09730750228014313,
            0.08496322733983144,
            0.06403124237432933,
        ],
    },
    "RetroMAE": {
        "x": ["0", "1000", "5000", "15000", "50000", "100000", "500000", "None"],
        "y": [
            42.16,
            42.4225,
            42.0,
            42.480000000000004,
            43.612500000000004,
            44.002500000000005,
            44.897499999999994,
            45.39,
        ],
        "std": [
            0,
            0.3800904497616312,
            0.18854707634964668,
            0.1083974169433934,
            0.15417117110536574,
            0.14024532077755947,
            0.044370598373248804,
            0.030822070014845666,
        ],
    },
    "SIMCSE-Small-E-Czech": {
        "x": ["0", "1000", "5000", "15000", "50000", "100000", "500000", "None"],
        "y": [39.20, 39.0125, 39.07, 40.0225, 42.1675, 42.495000000000005, 43.8725, 44.7],
        "std": [
            0,
            0.12871965661856113,
            0.16186414056238718,
            0.24406710142909444,
            0.15122417134836466,
            0.07158910531638205,
            0.06571719714047305,
            0.05244044240850765,
        ],
    },
    "Small-E-Czech": {
        "x": ["0", "1000", "5000", "15000", "50000", "100000", "500000", "None"],
        "y": [37.31, 38.3675, 39.2875, 39.365, 40.7525, 41.199999999999996, 42.86749999999999, 43.905],
        "std": [
            0,
            0.643365176241299,
            0.4552128622963108,
            0.3457238782612505,
            0.15481844205390963,
            0.12083045973594433,
            0.143418095092635,
            0.05408326913195959,
        ],
    },
}


model_names = list(data.keys())

train_sizes = [x if x != "None" else "1400000" for x in data[model_names[0]]["x"]]
train_sizes = [int(x) // 1000 for x in train_sizes]
# train_sizes = list(map(format_number_with_unicode_spaces, train_sizes))
train_sizes = [str(x) + "k" for x in train_sizes]

mean_results = np.array([data[model_name]["y"] for model_name in model_names])
std_devs = np.array([data[model_name]["std"] for model_name in model_names])


# In[25]:


# Create a figure and axis
fig, ax = plt.subplots()

# Plot each model's line and shaded area for standard deviation
for i, model_name in enumerate(model_names):
    line = ax.plot(train_sizes, mean_results[i], marker="o", linestyle="-", label=model_name)
    ax.fill_between(
        train_sizes, mean_results[i] - std_devs[i], mean_results[i] + std_devs[i], color=line[0].get_color(), alpha=0.2
    )

# Add labels and title
ax.set_xlabel("Training Data Size")
ax.set_ylabel("DaReCzech [P@10]")
# ax.set_title('Model Performance vs. Training Data Size')

# Add legend
ax.legend()

# Show the plot
# plt.show()
matplotlib.rcParams.update({"font.size": 8})

plt.savefig("finetuning_datasize.pdf")


# In[ ]:
