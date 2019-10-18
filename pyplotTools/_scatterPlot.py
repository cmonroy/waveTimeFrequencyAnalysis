"""
scatter plot colored by density (kde)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn




def scatterPlot(df , x, y , ax = None, x_y = False , title = None, meanCov = None, **kwargs) :
    """
    Scatter plot, with additional option compared to pandas.plot(kind="scatter")
    """

    if ax is None :
        fig ,ax = plt.subplots()

    df.plot(ax=ax,x=x, y=y, kind = "scatter", **kwargs)

    _x = df.loc[:,x]
    _y = df.loc[:,y]

    displayMeanCov(df,_x,_x,meanCov,ax)

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



def density_scatter( x , y, ax = None, sort = True, bins = 20, scale = None, interpolation = "linear", x_y = False,  **kwargs )   :
    """
    Scatter plot colored by 2d histogram
    """
    if ax is None :
        fig , ax = plt.subplots()
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = interpolation, bounds_error = False, fill_value = np.nan )
    if scale is not None :
        z = scale(z)

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]


    if x_y :
        add_x_y( x, y,  ax )

    ax.scatter( x, y, c=z, edgecolor = "", **kwargs )
    return ax


def add_x_y( x,y, ax ) :
    minMax = [min(x.min(),y.min()),max(x.max(),y.max())]
    ax.plot(minMax , minMax , "-" )


def displayMeanCov(df,x,y, meanCov,ax):
    if meanCov is not None :
        if meanCov is True:
            mean = np.mean((df.loc[:,y] / df.loc[:,x]))
            cov = np.std((df.loc[:,y] / df.loc[:,x])) / mean
            mean -= 1
            ax.text( 0.8 , 0.2 ,  "mean : {:.1%}\nCOV : {:.1%}".format(mean , cov) , transform=ax.transAxes ) # verticalalignment='center'

        elif meanCov == "abs_mean_std" :
            mean = np.mean((df.loc[:,y] - df.loc[:,x]))
            std = np.std((df.loc[:,y] - df.loc[:,x]))
            ax.text( 0.8 , 0.2 ,  "mean : {:.2f}\nSTD : {:.2f}".format(mean , std) , transform=ax.transAxes ) # verticalalignment='center'


if "__main__" == __name__ :

    x = np.random.normal(size=100000)
    y = x * 3 + np.random.normal(size=100000)
    density_scatter( x, y, bins = [30,30] )
