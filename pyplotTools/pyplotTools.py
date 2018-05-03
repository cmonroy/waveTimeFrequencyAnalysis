import matplotlib

import itertools
colorCycle = ('b', 'r', 'c' , 'm', 'y' , 'k', 'g')
def newColorIterator() :
    return itertools.cycle(colorCycle)


markerCycle = ('o', 'v', "s", '*' , 'D')
def newMarkerIterator() :
    return itertools.cycle(markerCycle)


linestyleCycle = ('-', '--', '-.', ':')
def newLinestyleIterator() :
    return itertools.cycle(linestyleCycle)


def pyplotLegend(plt):
    ax = plt.get_axes()
    handles, labels =  ax[0].get_legend_handles_labels()
    uniqueLabels = sorted(list(set(labels )))
    uniqueHandles = [handles[labels.index(l)] for l in uniqueLabels ]
    return uniqueHandles, uniqueLabels


def autoscale_y(ax,margin=0.1):
    """This function rescales the y-axis based on the data that is visible given the current xlim of the axis.
    ax -- a matplotlib axes object
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_bottom_top(line):
        xd = line.get_xdata()
        yd = line.get_ydata()
        lo,hi = ax.get_xlim()
        y_displayed = yd[((xd>lo) & (xd<hi))]
        h = np.max(y_displayed) - np.min(y_displayed)
        bot = np.min(y_displayed)-margin*h
        top = np.max(y_displayed)+margin*h
        return bot,top

    lines = ax.get_lines()
    bot,top = np.inf, -np.inf

    for line in lines:
        new_bot, new_top = get_bottom_top(line)
        if new_bot < bot: bot = new_bot
        if new_top > top: top = new_top

    ax.set_ylim(bot,top)