import numpy as np
from matplotlib import pyplot as plt


def qqplot(data, dist, x_y = True, ax=None, **kwargs):
    """
    Standard qqpLot
    """
    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)
    fig, ax = plt.subplots()
    ax.plot( dist.ppf( np.arange(1,n+1)/(n+1) ), np.sort(data), "+" )

    if x_y is True:
        ax.plot( [np.min(data), np.max(data)], [np.min(data), np.max(data)], "-" )

    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Emprical")
    ax.set_title("QQ plot")
    return ax


def qqplot2(data_1, data_2, data_1_label=None, data_2_label=None, ax=None, x_y = True, marker = "+", **kwargs):
    """
    QQ plot for two set of data (scatter plot of sorted values)
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot( np.sort(data_1), np.sort(data_2), marker = marker, **kwargs )

    if x_y is True :
        ax.plot( [np.min(data_1), np.max(data_1)], [np.min(data_1), np.max(data_1)], "-" )

    if data_1_label is not None :
        ax.set_xlabel(data_1_label)

    if data_1_label is not None :
        ax.set_ylabel(data_2_label)

    ax.set_title("QQ plot")
    return ax
