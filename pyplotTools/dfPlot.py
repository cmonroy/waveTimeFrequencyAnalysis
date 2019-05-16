"""

   Additional plotting tools for pandas DataFrame

"""


from __future__ import print_function
from matplotlib import pyplot as plt

from matplotlib.colors import Normalize

import numpy as np
import pandas as pd
import xarray as xa


def centerToEdge(array):
    dx = array[1:] - array[:-1]
    if (abs(dx / dx[0] - 1) < 0.01).all():
        return np.append(array - 0.5*dx[0], array[-1] + 0.5*dx[0])
    else:
        print("Can not find edge from center if bins are not considered evenly spaced")
        raise(ValueError())




def compareSurfaceAndMarginals(  df1, df2, surfaceArgs, cumulative = True, name1 = None, name2 = None, kwargs_l1= {}, kwargs_l2 = {}):

    fig, axList = plt.subplots( nrows = 2, ncols = 2 )

    dfSurface(df1, ax = axList[1,0], **surfaceArgs)
    axList[1,0].set_xlabel( axList[1,0].get_xlabel() +  "\n" + name1  )
    dfSurface(df2, ax = axList[0,1], **surfaceArgs)
    axList[0,1].set_xlabel( axList[0,1].get_xlabel() +  "\n" + name2  )


    for df, name, kwargs in [ (df1, name1, kwargs_l1), (df2, name2, kwargs_l2)]:
        if not cumulative :
            axList[1,1].barh( df.index, df.sum(axis=1), alpha = 0.5 , label = name)
            axList[0,0].bar( df.columns, df.sum(axis=0), alpha = 0.5, label = name )
        else:
            hsCum = np.add.accumulate( df.sum(axis=1)[::-1] )[::-1]  # 1-np.add.accumulate( df.sum(axis=1) )
            axList[1,1].plot(hsCum,  hsCum.index, label = name, **kwargs )
            axList[1,1].set_xscale("log")
            tzCum = np.add.accumulate( df.sum(axis=0)[::-1] )[::-1]  # 1-np.add.accumulate( df.sum(axis=0) )
            axList[0,0].plot(tzCum.index, tzCum, "o", label = name, **kwargs )
            axList[0,0].set_yscale("log")

    axList[0,0].legend()
    ymin = min( df1.index.min() , df2.index.min())
    ymax = max( df1.index.max() , df2.index.max())

    axList[1,1].set_ylim([ymin, ymax])
    axList[1,0].set_ylim([ymin, ymax])

    xmin = min( df1.columns.min() , df2.columns.min())
    xmax = max( df1.columns.max() , df2.columns.max())

    axList[0,0].set_xlim([xmin, xmax])
    axList[0,1].set_xlim([xmin, xmax])

    axList[0,0].set_ylabel( "Exceedance rate " + df.columns.name )
    axList[1,1].set_xlabel( "Exceedance rate " + df.index.name )
    plt.tight_layout()
    return fig

def dfSurfaceAndMarginals( df, surfaceArgs, cumulative = True ):
    fig, axList = plt.subplots( nrows = 2, ncols = 2 )

    dfSurface(df, ax = axList[1,0], **surfaceArgs)
    if not cumulative :
        axList[1,1].barh( df.index, df.sum(axis=1), alpha = 0.5 )
        axList[0,0].bar( df.columns, df.sum(axis=0), alpha = 0.5 )
    else:
        hsCum = 1-np.add.accumulate( df.sum(axis=1) )
        axList[1,1].plot(hsCum,  hsCum.index, "o" )
        axList[1,1].set_xscale("log")
        tzCum = 1-np.add.accumulate( df.sum(axis=0) )
        axList[0,0].plot(tzCum.index, tzCum, "o" )
        axList[0,0].set_yscale("log")

    axList[1,1].set_ylim([df.index.min(), df.index.max()])
    axList[1,0].set_ylim([df.index.min(), df.index.max()])

    axList[0,0].set_ylabel( "Exceedance rate " + df.columns.name )
    axList[1,1].set_ylabel( "Exceedance rate " + df.index.name )

    axList[0,0].set_xlim([df.columns.min(), df.columns.max()])
    axList[0,1].set_xlim([df.columns.min(), df.columns.max()])

    axList[0,1].axis("off")
    plt.tight_layout()
    return fig


def dfSurface(df, ax=None, nbColors=200, interpolate=True,
              polar=False, polarConvention="seakeeping", colorbar=False, scale = None, vmin = None, vmax = None, **kwargs):
    """Surface plot from pandas.DataFrame
    
    Parameters
    ----------
    df: pandas.DataFrame or xarray.DataArray
    
        Dataframe formmated as follow:
        * columns : x or theta values
        * index : y or r values
        * data = data
        
        DataArray of dimension 2 formmated as follow:
        * data = data
        * dims : x and y values

    ax: matplotlib.axes._subplots.AxesSubplot
        Specify existing axis to plot
        
    nbColors: int, default 200
        Number of colorscale levels
        
    interpolate: bool, default True
        * if True, data are considered as node value and interpolated in between
        * if False, data are considered as center cell value  => similar to sns.heatmap
        
    colorbar: bool, default False
        Specify is colobar should be plotted
        
    **kwargs:
        Arguments applied to matplotlib.axes.Axes.contourf
    """
    
    if type(df)==xa.DataArray: raise NotImplementedError
        
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=polar)
        if polar:
            if polarConvention == "seakeeping":
                ax.set_theta_zero_location("S")
                ax.set_theta_direction(1)
            elif polarConvention == "geo":
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(1)
            
            if df.columns.astype(float).max()>2*np.pi:
                df.columns = [np.deg2rad(i) for i in df.columns.astype(float)]
                
    if scale is not None:
        val = scale(df.values)
    else :
        val = df.values

    if interpolate:
        cax = ax.contourf(df.columns.astype(float), df.index, val, levels=nbColors, vmin=vmin, vmax=vmax, **kwargs)
    else:
        try:
            cax = ax.pcolormesh(centerToEdge(df.columns.astype(float)), centerToEdge(df.index), val, vmin=vmin, vmax=vmax, **kwargs)
        except ValueError:
            raise(Exception("Index is not evenly spaced, try with interpolate = True"))

    # Add x and y label if contains in the dataFrame
    if df.columns.name is not None:
        ax.set_xlabel(df.columns.name)
    if df.index.name is not None:
        ax.set_ylabel(df.index.name)

    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if colorbar:
        cbar = ax.get_figure().colorbar(cax)
        cbar.set_clim(vmin=vmin, vmax=vmax)

    return ax


def dfIsoContour(df, ax=None, polar=False, polarConvention="seakeeping", inline=True, clabels=None, **kwargs):
    """Iso contour plot from pandas.DataFrame
    
    Parameters
    ----------
    df: pandas.DataFrame
        Dataframe formmated as follow:
            
        * index : y or theta values
        * columns : x or theta values
        * data = data
        
    ax: matplotlib.axes._subplots.AxesSubplot
        Specify existing axis to plot
        
    clabels: list
        Custom contour labels

    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, polar=polar)
        if polar:
            if polarConvention == "seakeeping":
                ax.set_theta_zero_location("S")
                ax.set_theta_direction(1)
            elif polarConvention == "geo":
                ax.set_theta_zero_location("N")
                ax.set_theta_direction(1)

    cax = ax.contour(df.columns.astype(float), df.index, df.values, **kwargs)

    if inline:
        if clabels is not None:
            fmt = {}
            for l, s in zip(cax.levels, clabels):
                fmt[l] = s
            ax.clabel(cax, cax.levels, inline=True, fmt=fmt, fontsize=10)
        else:
            ax.clabel(cax, inline=1, fontsize=10,  fmt=r"%1.1f")

#    ax.legend()
    # Add x and y label if contains in the dataFrame
    if df.columns.name is not None:
        ax.set_xlabel(df.columns.name)
    if df.index.name is not None:
        ax.set_ylabel(df.index.name)
    return ax


def dfSlider(dfList, labels=None, ax=None, display=True, **kwargs):
    """ Interactive 2D plots, with slider to select the frame to display

    Column is used as x axis
    Index is used as frame/time (which is selected with the slider)

    :param dfList: List of DataFrame to animate
    :param labels: labels default = 1,2,3...
    :param ax: Use existing ax if provided
    :param display: display the results (wait for next show() otherwise)

    :return:  ax

    """

    print("Preparing interactive plot")
    from matplotlib.widgets import Slider
    import numpy as np

    # Make compatible with single dataFrame input
    if type(dfList) is not list:
        dfList = [dfList]
    if labels is None:
        labels = [i for i in range(len(dfList))]

    if ax is None:
        fig, ax = plt.subplots()

    plt.subplots_adjust(bottom=0.20)
    ax.grid(True)

    a0 = 0
    global currentValue
    currentValue = dfList[0].index[a0]

    lList = []
    for idf, df in enumerate(dfList):
        l, = ax.plot(df.columns, df.iloc[a0, :], lw=2, label=labels[idf], **kwargs)
        lList.append(l)

    ax.legend(loc=2)

    df = dfList[0]
    ax.set_title(df.index[0])

    tmin = min([min(df.columns) for df in dfList])
    tmax = max([max(df.columns) for df in dfList])
    ymin = min([df.min().min() for df in dfList])
    ymax = max([df.max().max() for df in dfList])

    #plt.axis( [tmin, tmax, ymin , ymax ] )
    ax.set_xlim([tmin, tmax])
    ax.set_ylim([ymin, ymax])

    axTime = plt.axes([0.15, 0.10, 0.75, 0.03], facecolor='lightgoldenrodyellow')
    sTime = Slider(axTime, 'Time', df.index[0], df.index[-1], valinit=a0)

    def update(val):
        global currentValue
        t = []
        for i, df in enumerate(dfList):
            itime = np.argmin(np.abs(df.index.values - sTime.val))
            lList[i].set_ydata(df.iloc[itime, :])
            t.append("{:.1f}".format(df.index[itime]))
            currentValue = val
        ax.set_title(" ".join(t))
        ax.get_figure().canvas.draw_idle()

    update(currentValue)

    def scroll(event):
        global currentValue
        s = 0
        if event.button == 'down' and currentValue < tmax:
            s = +1
        elif event.button == 'up' and currentValue > tmin:
            s = -1
        dt = dfList[0].index[1]-dfList[0].index[0]
        sTime.set_val(currentValue + s*dt)

    ax.get_figure().canvas.mpl_connect('scroll_event', scroll)
    sTime.on_changed(update)

    if display:
        plt.show()

    # Return ax, so that it can be futher customized ( label... )
    return ax


def dfAnimate(df, movieName=None, nShaddow=0, xRatio=1.0, rate=1, xlim=None, ylim=None, xlabel="x(m)", ylabel="Elevation(m)", codec="libx264", **kwargs):
    """
       Animate a dataFrame where time is the index, and columns are the "spatial" position
    """

    from matplotlib import animation

    print("Making animation file : ", movieName)

    global pause
    pause = False

    def onClick(event):
        global pause
        pause ^= True

    nShaddow = max(1, nShaddow)

    fig, ax = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ls = []
    for i in range(nShaddow):
        if i == 0:
            color = "black"
        else:
            color = "blue"
        ltemp,  = ax.plot([], [], lw=1, alpha=1-i*1./nShaddow, color=color, **kwargs)
        ls.append(ltemp)

    xVal = df.columns.astype(float)

    ax.grid(True)

    if xlim:
        ax.set_xlim(xlim)
    else:
        ax.set_xlim(min(xVal), max(xVal))

    if ylim:
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(df.min().min(), df.max().max())

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    def run(itime):
        ax.set_title("{:.1f}s".format(df.index[itime*rate]))
        for s in range(nShaddow):
            if not pause:
                if itime > s:
                    ls[s].set_data(xVal, df.iloc[rate*(itime - s), :])
        return ls

    ani = animation.FuncAnimation(fig, run, range(len(df)), blit=True, interval=30, repeat=False)

    if movieName is None:
        plt.show()
    else:
        mywriter = animation.FFMpegWriter(fps=25, codec=codec)
        ani.save(movieName + '.mp4', writer=mywriter)


def testSlider():
    from Pluto.Spectral import Wif
    wif = Wif.Jonswap(Hs=2.0, Tp=10.0, Gamma=1.0, Heading=180.)
    #wif = Wif( r"D:\Etudes\HOS\edw\my_10_rcw.wif" )
    #wif.position( -400. , 0. )
    df = wif.wave2DC(tmin=0, tmax=+200, dt=0.2, xmin=0., xmax=+800, dx=4.0, speed=0.0)
    df2 = wif.wave2DC(tmin=0, tmax=+200, dt=0.2, xmin=100., xmax=+800, dx=4.0, speed=0.0)
    #dfAnimate(df, nShaddow = 5, xRatio= 1.0, rate = 1)
    dfSlider([df, df2])


def testSurfacePlot():
    df = pd.DataFrame(index=np.linspace(0, 0.5, 100), columns=np.linspace(0, 2*np.pi, 50), dtype="float")
    df.loc[:, :] = 1
    for i in range(len(df.columns)):
        df.iloc[:, i] = np.sin(df.columns[i]) * df.index.values
    ax = dfSurface(df, polar=True,  interpolate=True, polarConvention="geo", colorbar=True, vmin = 0.2, vmax = 0.3)
    ax = dfIsoContour(df, levels=[0.0, 0.5], ax=ax)
    plt.show()


if __name__ == "__main__":

    print("Test")
    testSurfacePlot()
#    testSlider()
