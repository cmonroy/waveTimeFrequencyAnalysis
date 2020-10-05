import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
import matplotlib

def drawMap(ax=None, projection=None, central_longitude=0.0, lcolor='grey', scolor=None):
    from cartopy import crs as ccrs, feature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    
    if projection is None: projection = ccrs.PlateCarree(central_longitude=central_longitude)
    if ax is None: fig, ax = plt.subplots( figsize = [12,6],  subplot_kw={'projection':projection })
    
    ax.coastlines()
    ax.add_feature(feature.LAND, facecolor=lcolor)
    if scolor is not None: ax.add_feature(feature.OCEAN, facecolor=scolor)
    ax.set_xticks( np.arange(-180, 210 , 30), crs=ccrs.PlateCarree())
    ax.set_yticks([-90, -60, -30, 0, 30, 60, 90], crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    
    return ax

def drawRoute(pathPoint, var=None, label = None,  ax=None, central_longitude=0.0, zoom = "full" , markersize = 5, lcolor='grey', scolor=None, cbar = False, **kwargs):
    """Draw route on earth map


    Parameters
    ----------
    pathPoint : List, array of pd.DataFrame
        Path point to plotted.
        If dataFrame, should have "lat" and "lon" columns

    var : str, optional
        Columns to use to color path point. The default is None.
    label : TYPE, optional
        DESCRIPTION. The default is None.
    ax : matplotlib "axe", optional
        Where to plot. The default is None.
    central_longitude : float, optional
        central_longitude. The default is 0.0.
    zoom : str, optional
        DESCRIPTION. The default is "full".
    markersize : float, optional
        Marker size. The default is 5.
    lcolor : str, optional
        Color of land areas. The default is "grey".
    scolor : str, optional
        Color of sea/ocean areas. The default is None.
    cbar : bool, optional
        Add colorbar. The default is False.
    **kwargs : Any
        Keyword arguments passed to .plot().

    Returns
    -------
    ax :
        The "axe"

    """

    from cartopy import crs as ccrs, feature
    projection = ccrs.PlateCarree(central_longitude=central_longitude)
    if ax is None:
        ax = drawMap(projection=projection, central_longitude=central_longitude, lcolor=lcolor, scolor=scolor)

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
            
            if cbar:
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
                cbar = ax.get_figure().colorbar(sm, ax=ax, orientation='vertical',fraction=0.046, pad=0.04)
                cbar.set_label(var, rotation=90)
            
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

def animRoute(pathPoint, var=None, ax=None, central_longitude=0.0, zoom = "full" , markersize = 15, mcolor='b', lcolor='grey', scolor=None, every=1):
    """Animate route on earth map


    Parameters
    ----------
    pathPoint : pd.DataFrame
        Path point to plotted.
        Mandatory columns : "lat" and "lon".
        Optional columns : "time" and var.
    var : str, optional
        Columns to use to color path point. The default is None.
    ax : matplotlib "axe", optional
        Where to plot. The default is None.
    central_longitude : float, optional
        central_longitude. The default is 0.0.
    zoom : str, optional
        DESCRIPTION. The default is "full".
    markersize : float, optional
        Marker size. The default is 5.
    mcolor : str, optional
        Marker color. The default is 'b'.
    lcolor : str, optional
        Color of land areas. The default is "grey".
    scolor : str, optional
        Color of sea/ocean areas. The default is None.
    every : int, optional
        Integer defining animation output rate. The default is 1.

    Returns
    -------
    anim :
        The "animation". Animation can then be saved with the following command :
        anim.save(path, writer=writer)

    """
    import matplotlib.animation as animation
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800)
        
    ax = drawMap(central_longitude=central_longitude,lcolor=lcolor, scolor=scolor)
    if zoom.lower() == "full" :
        ax.set_global()
        ax.set_xlim((-180., 180.))
        ax.set_ylim((-90.,  90.))
    point = ax.plot(0, 0, color=mcolor, markersize=markersize)[0]
    if var is not None:
        cmap = matplotlib.cm.get_cmap('viridis')
        vmin = mt.floor(pathPoint.loc[:,var].min())
        vmax = mt.ceil(pathPoint.loc[:,var].max())
        # norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        line = ax.scatter(0, 0, color=mcolor, marker=',', s=10,cmap='viridis',vmin=vmin,vmax=vmax)
    else:
        # line = m.plot(0, 0, color='b', ls=':', lw=5)[0]
        line = ax.scatter(0, 0, color=mcolor, marker='.', s=5)
        
    def init():
        point.set_data([], [])
        return point, line
        
    def animate(i):
        j = i*every
        lat = pathPoint.loc[j,'lat']
        lon = pathPoint.loc[j,'lon']
        angle = 360.-pathPoint.loc[j,'Dir']
        point.set_data(lon, lat)
        point.set_marker((3, 0, angle))
        
        lat_s = pathPoint.loc[:j,'lat'][::every*2]
        lon_s = pathPoint.loc[:j,'lon'][::every*2]
        if var is not None:
            var_s = pathPoint.loc[:j,var][::every*2]
            line.set_offsets(np.array([lon_s,lat_s]).T)
            line.set_array(var_s.values)
        else:
            # line.set_data(lon_s,lat_s)
            line.set_offsets(np.array([lon_s,lat_s]).T)
        
        if "time" in pathPoint.columns: plt.title(pathPoint.time[j])
        else: plt.title(pathPoint.index[j])
        
        return point, line
            
    anim = animation.FuncAnimation(ax.get_figure(), animate, frames=int(pathPoint.shape[0]/every), init_func=init, repeat=True, blit=True)
        
    return anim, writer

def mapPlot(  dfMap , ax=None, isoLevel = None , central_longitude=0.0  , vmin=None , vmax=None, cmap = "cividis", color_bar = False) :
    """
    Plot scalar field map. (same as mapPlot, but based on Cartopy)

    dfMap.index  => longitude
    dfMap.columns => latitude
    dfMap.data => scalar to plot
    """
    
    from cartopy import crs as ccrs, feature
    projection = ccrs.PlateCarree(central_longitude=central_longitude)
    if ax is None:
        ax = drawMap(projection=projection, central_longitude=central_longitude, lcolor=lcolor, scolor=scolor)
    
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