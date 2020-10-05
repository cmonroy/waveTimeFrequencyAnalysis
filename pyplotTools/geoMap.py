import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib

def drawRoute(pathPoint, var=None, label = None,  ax=None, central_longitude=0.0, zoom = "full" , markersize = 5, **kwargs):

    """
    Same as drawRoute, but based on cartopy

    pathPoint can be :
        - a list if (lat, lon) tuple
        - a DataFrame with "lon" and "lat" columns
        - a array

    """
    from cartopy import crs as ccrs, feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    projection = ccrs.PlateCarree(central_longitude=central_longitude)
    if ax is None:
        fig, ax = plt.subplots( figsize = [12,6],  subplot_kw={'projection':projection }  )
        ax.coastlines()
        ax.add_feature(feature.LAND, facecolor="gray")
        ax.set_xticks( np.arange(-180, 210 , 30), crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())

    if type(pathPoint) == list:  # List of [long, lat] tuple
        for iPoint in range(len(pathPoint)):
            lat, long = pathPoint[iPoint]
            ax.plot(long , lat,  "bo", markersize = markersize, **kwargs)

    elif type(pathPoint) == pd.DataFrame:
        if var is not None:
            # Draw route colored by field value
            cmap = matplotlib.cm.get_cmap('viridis')
            norm = matplotlib.colors.Normalize(vmin=np.min(pathPoint.loc[:,var]), vmax=np.max(pathPoint.loc[:,var]))
            ax.scatter(pathPoint["lon"], pathPoint["lat"],  s = markersize , c = cmap(norm(pathPoint.loc[:, var].values)), **kwargs)
        else:
            ax.plot(pathPoint["lon"], pathPoint["lat"], "bo", markersize = markersize, **kwargs)

        if label is not None :
            for _, row in pathPoint.iterrows():
                ax.text(row.lon - 3, row.lat - 3, row.loc[label],  horizontalalignment='right',  transform=projection, bbox=dict(boxstyle="square", fc="w"))

    else:  # Array
        ax.plot(pathPoint[:, 1], pathPoint[:, 0],  "bo", markersize = markersize, **kwargs)

    if zoom.lower() == "full" :
        ax.set_global()
        ax.set_xlim((-180., 180.))
        ax.set_ylim((-90.,  90.))
    return ax



def mapPlot(  dfMap , ax=None, isoLevel = None , central_longitude=0.0  , vmin=None , vmax=None, cmap = "cividis", color_bar = False) :
    from cartopy import crs as ccrs, feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    """
    Plot scalar field map. (same as mapPlot, but based on Cartopy)

    dfMap.index  => longitude
    dfMap.columns => latitude
    dfMap.data => scalar to plot
    """

    if ax is None:
        fig, ax = plt.subplots( figsize = [12,6],  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=central_longitude)}  )
        ax.coastlines()
        ax.add_feature(feature.LAND, facecolor="gray")
        ax.set_xticks( np.arange(-180, 210 , 30), crs=ccrs.PlateCarree())
        ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter())

    if vmin is None : vmin = np.min( dfMap.values[  (~np.isnan(dfMap.values)) ] )
    if vmax is None : vmax = np.max( dfMap.values[  (~np.isnan(dfMap.values)) ] )

    ax.coastlines()
    ax.add_feature(feature.LAND, facecolor="gray")
    cf = ax.contourf(dfMap.index.values, dfMap.columns.values, np.transpose(dfMap.values), 60,  cmap = cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree() )

    if color_bar :
        if vmin is not None and vmax is not None : extend = "both"
        elif vmin is None : extend = "max"
        else : extend = "min"
        cbar = plt.colorbar( ScalarMappable(norm=cf.norm, cmap=cf.cmap), extend = extend)
        if isinstance(color_bar , str) :
            cbar.set_label( color_bar )

    return ax