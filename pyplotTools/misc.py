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

def autoscale_xy(ax,axis='y',margin=0.1):
    """This function rescales the x-axis or y-axis based on the data that is visible on the other axis.
    ax -- a matplotlib axes object
    axis -- axis to rescale ('x' or 'y')
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_boundaries(xd,yd,axis):
        if axis=='x':
            bmin,bmax = ax.get_ylim()
            displayed = xd[((yd>bmin) & (yd<bmax))]
        elif axis=='y':
            bmin,bmax = ax.get_xlim()
            displayed = yd[((xd>bmin) & (xd<bmax))]
        h = np.max(displayed) - np.min(displayed)
        cmin = np.min(displayed)-margin*h
        cmax = np.max(displayed)+margin*h
        return cmin,cmax

    cmin,cmax = np.inf, -np.inf
    
    #For lines
    for line in ax.get_lines():
        xd = line.get_xdata(orig=False)
        yd = line.get_ydata(orig=False)
        new_min, new_max = get_boundaries(xd,yd,axis=axis)
        if new_min < cmin: cmin = new_min
        if new_max > cmax: cmax = new_max

    #For other collection (scatter)
    for col in ax.collections:
        xd = col.get_offsets().data[:,0]
        yd = col.get_offsets().data[:,1]
        new_min, new_max = get_boundaries(xd,yd,axis=axis)
        if new_min < cmin: cmin = new_min
        if new_max > cmax: cmax = new_max

    if   axis=='x': ax.set_xlim(cmin,cmax)
    elif axis=='y': ax.set_ylim(cmin,cmax)