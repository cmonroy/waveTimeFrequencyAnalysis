"""
scatter plot colored by density (kde)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.interpolate import interpn
from scipy.stats import linregress




def scatterPlot(df , x, y , ax = None, x_y = False , title = None, meanCov = None, **kwargs) :
    """
    Scatter plot, with additional option compared to pandas.plot(kind="scatter")
    """

    if ax is None :
        fig ,ax = plt.subplots()

    df.plot(ax=ax,x=x, y=y, kind = "scatter", **kwargs)

    _x = df.loc[:,x]
    _y = df.loc[:,y]

    displayMeanCov( x,y,meanCov,ax)

    if x_y is True :
        add_x_y(_x,_y,ax)

    return ax


def kde_scatter( x , y, ax = None, sort = True , lib_kde = "scipy", **kwargs )   :
    """
    Scatter plot colored by kde
    """
    if ax is None :
        fig , ax = plt.subplots()

    # Calculate the point density
    if lib_kde == "scipy" :
        from scipy.stats import gaussian_kde
        xy = np.vstack([x,y])
        z = gaussian_kde(xy)(xy)

    else :
        #With sklearn
        from sklearn.neighbors.kde import KernelDensity
        xy = np.vstack([x,y]).T
        a = KernelDensity().fit(X = xy)
        z = a.score_samples( xy  )


    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, **kwargs )
    return ax



def density_scatter( x , y, ax = None, sort = True, bins = 20, scale = None, interpolation = "linear",
                    x_y = False, cbar = False, range = None, **kwargs )   :
    """Scatter plot colored by density (2d histogram)
    To be prefered over kde_scatter when number of point in big (kde is then quite heavy to compute)
    

    Parameters
    ----------
    x : np.ndarray
        X data 
    y : np.ndarray
        Y data 
    ax : matplotlib.axes, optional
        Where to plot the figure. The default is None.
    sort : TYPE, optional
        DESCRIPTION. The default is True.
    bins : TYPE, optional
        DESCRIPTION. The default is 20.
    scale : function, optional
        Color map scale. The default is None.
    interpolation : "linear" or "nearest" or "splinef2d", optional
        How to interpolate colors in the 2D histogram. The default is "linear".
    x_y : bool, optional
        If True x=y line is plotted. The default is False.
    cbar : bool, optional
        Color bar. The default is False.
    **kwargs : *
        optional arguments passed to plt.scatter()

    Returns
    -------
    ax : matplotlib.axes
        ax

    """
    
    if ax is None :
        fig , ax = plt.subplots()


    data , x_e, y_e = np.histogram2d( x, y, bins = bins, density = True)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , 
                   method = interpolation, bounds_error = False)
    
    edges_id = np.isnan(z)
    z[ edges_id ] = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T[edges_id] , 
                             method = "nearest", bounds_error = False, fill_value = None)
    
    
    if scale is not None :
        z = scale(z)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    if x_y :
        add_x_y( x, y,  ax )

    ax.scatter( x, y, c=z, edgecolor = None, **kwargs )
    
    #Add color baz : 
    if cbar :
        norm = Normalize(vmin = np.min(z), vmax = np.max(z))
        cbar = ax.get_figure().colorbar(cm.ScalarMappable(norm = norm), ax=ax)
        cbar.ax.set_ylabel('Density')
        
    if hasattr( x , "name" ) :
        ax.set_xlabel(x.name)
    if hasattr( y , "name" ) :
        ax.set_ylabel(y.name)

    return ax


def add_x_y( x,y, ax ) :
    minMax = [min(x.min(),y.min()),max(x.max(),y.max())]
    ax.plot(minMax , minMax , "-" )


def add_linregress( x, y, ax, text = True ):
    
    minMax = np.array( [ min(x), max(x) ] )
    lreg = linregress(x, y)
    
    label = f"y = {lreg.slope:.1f} x {lreg.intercept:+.1f}"
    ax.plot( minMax , lreg.slope * minMax  + lreg.intercept, "-", label = label )
    ax.legend()
    
    return ax
  


def displayMeanCov(x,y, meanCov,ax):
    if meanCov is not None :
        if meanCov is True:
            mean = np.mean((y / x))
            cov = np.std((y / x)) / mean
            mean -= 1.
            ax.text( 0.8 , 0.2 ,  "mean : {:.1%}\nCOV : {:.1%}".format(mean , cov) , transform=ax.transAxes ) # verticalalignment='center'

        elif meanCov == "abs_mean_std" :
            mean = np.mean((y - x))
            std = np.std((y - x))
            ax.text( 0.8 , 0.2 ,  "mean : {:.2f}\nSTD : {:.2f}".format(mean , std) , transform=ax.transAxes ) # verticalalignment='center'
            

if "__main__" == __name__ :

    x = np.random.normal(size=10000)
    y = x * 3 + np.random.normal(size=10000)
    ax = density_scatter( x, y, bins = [30,30] )
    
    add_linregress(x,y,ax=ax)
    
    
    
    
    
    
