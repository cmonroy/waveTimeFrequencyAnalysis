from .dfPlot import dfSlider, dfAnimate, dfSurface, dfIsoContour
from .misc import newColorIterator, markerCycle, newMarkerIterator, linestyleCycle, colorCycle
from .misc import newLinestyleIterator, pyplotLegend, autoscale_xy, uniqueLegend
from .misc import getColorMappable, getAngleColorMappable, hexa_to_rgb, rgb_to_hexa, negativeColor
from .surfacePlot import mapFunction
from ._scatterPlot import kde_scatter, density_scatter, scatterPlot, add_linregress,add_x_y, displayMeanCov
from .mplZoom import ZoomPan
from .meshPlot import plotMesh
from .statPlots import qqplot, qqplot2, distPlot, probN, probN_ci
from .addcopyfighandler import copyToClipboard_on
from .concatPlot import readImage, concatPlot
from .geoMap import mapPlot, drawRoute, animRoute, drawMap, standardLon, drawGws