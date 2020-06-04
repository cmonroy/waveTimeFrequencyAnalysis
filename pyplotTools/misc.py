import itertools
import numpy as np
colorCycle = ('b', 'r', 'c' , 'm', 'y' , 'k', 'g')
def newColorIterator() :
    return itertools.cycle(colorCycle)


markerCycle = ('o', 'v', "s", '*' , 'D')
def newMarkerIterator() :
    return itertools.cycle(markerCycle)


linestyleCycle = ('-', '--', '-.', ':')
def newLinestyleIterator() :
    return itertools.cycle(linestyleCycle)


def getAngleColorMappable( unit = "rad", cmap = "twilight" ):
    if "rad" in unit.lower() :
        vmax = 2*np.pi
    else :
        vmax = 360.
    return getColorMap( vmin = 0.0 , vmax = vmax , cmap = cmap )


def getColorMappable( vmin, vmax, cmap = "viridis" ):
    import matplotlib.colors as colors
    import matplotlib.cm as cm
    cNorm  = colors.Normalize( vmin=vmin, vmax=vmax)
    scalarMap = cm.ScalarMappable(norm=cNorm, cmap=cmap)
    return scalarMap


def pyplotLegend(plt=None,ax=None):
    if plt is not None :
        ax = plt.get_axes()[0]
    handles, labels =  ax.get_legend_handles_labels()
    uniqueLabels = sorted(list(set(labels )))
    uniqueHandles = [handles[labels.index(l)] for l in uniqueLabels ]
    return uniqueHandles, uniqueLabels


def uniqueLegend(ax, *args, **kwargs) :
    ax.legend( *pyplotLegend(ax=ax), *args, **kwargs )


def autoscale_xy(ax,axis='y',margin=0.1):
    """This function rescales the x-axis or y-axis based on the data that is visible on the other axis.
    ax -- a matplotlib axes object
    axis -- axis to rescale ('x' or 'y')
    margin -- the fraction of the total height of the y-data to pad the upper and lower ylims"""

    import numpy as np

    def get_boundaries(xd,yd,axis):
        if axis == 'x':
            bmin,bmax = ax.get_ylim()
            displayed = xd[((yd > bmin) & (yd < bmax))]
        elif axis == 'y':
            bmin,bmax = ax.get_xlim()
            displayed = yd[((xd > bmin) & (xd < bmax))]
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