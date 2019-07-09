"""
scatter plot colored by density (kde)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interpn

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
        maxP, minP = max( np.max(x), np.max(y) ),  min( np.min(x), np.min(y) )
        ax.plot(   [minP , maxP] , [minP , maxP]  )

    ax.scatter( x, y, c=z, edgecolor = "", **kwargs )
    return ax



if "__main__" == __name__ :

    x = np.random.normal(size=100000)
    y = x * 3 + np.random.normal(size=100000)
    density_scatter( x, y, bins = [30,30] )
