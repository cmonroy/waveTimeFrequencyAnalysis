from .dfPlot import dfSlider, dfAnimate, dfSurface, dfIsoContour
from .misc import newColorIterator, markerCycle, newMarkerIterator, linestyleCycle
from .misc import newLinestyleIterator, pyplotLegend, autoscale_xy, uniqueLegend
from .misc import getAngleColorMap
from .surfacePlot import mapFunction
from ._scatterPlot import kde_scatter, density_scatter, scatterPlot, add_linregress
from .mplZoom import ZoomPan
from .meshPlot import plotMesh
from .statPlots import qqplot, qqplot2, distPlot, probN, probN_ci
from .addcopyfighandler import copyToClipboard_on
from .concatPlot import readImage, concatPlot