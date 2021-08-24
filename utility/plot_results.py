# -*- coding: utf-8 -*-
# import numpy as np
from matplotlib import pyplot
# import matplotlib.pyplot as plt


def plot_comparison_hist(values, labels, colors, title, x_label, y_label):
    for i, item in enumerate(values):
        pyplot.hist(item, color=colors[i], bins=25, alpha=0.5, label=labels[i])
    pyplot.title(title)
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(loc='upper left')
    pyplot.savefig(f"./results/{title}.png")
    pyplot.close()
