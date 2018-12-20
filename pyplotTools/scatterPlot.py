"""
scatter plot colored by density (kde)
"""

import numpy as np
import matplotlib.pyplot as plt

def kde_scatter( x , y, ax = None, edgecolor = "", sort = True , **kwargs )   :
    from scipy.stats import gaussian_kde


    if ax is None :
        fig , ax = plt.subplots()

    # Calculate the point density
    print ('kde')
    xy = np.vstack([x,y])
    z = gaussian_kde(xy)(xy)

    """
    #With sklearn
    from sklearn.neighbors.kde import KernelDensity
    xy = np.vstack([x,y]).T
    a = KernelDensity().fit(X = xy)
    z = a.score_samples( xy  )
    """

    # Sort the points by density, so that the densest points are plotted last
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]

    ax.scatter( x, y, c=z, edgecolor=edgecolor, **kwargs )
    return ax



if "__main__" == __name__ :

    from scipy.stats import multivariate_normal
    t = multivariate_normal.rvs( mean = [0,0], size = 10000)
    kde_scatter( t[:,0], t[:,1] )
