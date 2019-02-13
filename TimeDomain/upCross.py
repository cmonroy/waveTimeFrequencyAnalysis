import pandas as pd
import numpy as np

def getUpCrossID( array, threshold ) :
    """Get the upcrossing indices

       Input :
               array : numpy 1D array
               threshold
       Output :
               upCrossing indexes (numpy 1D array of integer)

    """
    isCross = (array[:-1] <= threshold) & (array[1:] > threshold)
    return np.where( isCross )[0]


def getDownCrossID( array, threshold ) :
    """Get the downcrossing indices

       Input :
               array : numpy 1D array
               threshold
       Output :
               upCrossing indexes (numpy 1D array of integer)

    """
    isCross = (array[:-1] > threshold) & (array[1:] <= threshold)
    return np.where( isCross )[0]



def getPeaksBounds(se, threshold):
    """Get peaks, identified by the up and down crossing of a threshold

    """
    array = se.values
    up_ = getUpCrossID( array, threshold )
    down_ = getDownCrossID( array, threshold )

    if down_[0] < up_[0] :
        down_ = down_[1:]

    if len(down_) != len(up_):
        up_ = up_[:-1]

    return up_, down_



"""
#Numba works but does not accelerate a lot the calculation (replace dtype=int by dtype=int32)
from numba import jit, float64 , int64, int32 , int16
from numba.types import Tuple
@jit(  Tuple((float64[:], float64[:],int32[:],int32[:]))(float64[:] , int64[:]) , nopython = True  )
"""
def minMax(array, upCrossID) :
    """
       Return max and min and position between each cycle
    """
    minimumTime = np.empty( ( len(upCrossID)-1 ) , dtype = int)
    maximumTime = np.empty( ( len(upCrossID)-1 ) , dtype = int)
    for iPoint in range(len(upCrossID)-1) :
        minimumTime[iPoint] = upCrossID[iPoint] + array[ upCrossID[iPoint] : upCrossID[iPoint + 1] ].argmin()
        maximumTime[iPoint] = upCrossID[iPoint] + array[ upCrossID[iPoint] : upCrossID[iPoint + 1] ].argmax()
    minimum =  array[ minimumTime ]
    maximum =  array[ maximumTime ]
    return minimum , maximum , minimumTime , maximumTime


def upCrossMinMax( se, upCrossID = None , threshold = "mean" ) :
    """
       Perform the "whole" upcrossing analysis
    """

    array = se.values

    #Compute threshold if not given
    if threshold == "mean" :
       threshold = np.mean(array)


    #Compute upCrossing index if not given
    if upCrossID is None :
       upCrossID = getUpCrossID( array , threshold = threshold )

    if len(upCrossID) == 0 :
        return pd.DataFrame( data = { "Minimum" : [] , "Maximum" : [] ,
                                 "MinimumTime" : [] , "MaximumTime" : [] ,
                                 "upCrossTime" : [] , "Period": []  } )

    #Fill values
    periods = np.empty( ( len(upCrossID)-1 )  , dtype = type(se.index.dtype) )
    minimum , maximum , minimumTime , maximumTime = minMax( array , upCrossID )
    minimumTime = se.index[ minimumTime ]
    maximumTime = se.index[ maximumTime ]
    upCrossTime = se.index[ upCrossID[:-1]]
    periods = se.index[ upCrossID[1:]] - se.index[upCrossID[:-1]]
    return pd.DataFrame( data = { "Minimum" : minimum , "Maximum" : maximum ,
                                  "MinimumTime" : minimumTime , "MaximumTime" : maximumTime ,
                                  "upCrossTime" : upCrossTime , "Period": periods  } )



def plotUpCross( upCrossDf , ax = None, cycleLimits = False ) :
    """
       Plot time trace together with extracted maxima
    """
    from matplotlib import pyplot as plt
    if ax is None : fig , ax = plt.subplots()
    if cycleLimits :
        max_ = upCrossDf.Maximum.max()
        min_ = upCrossDf.Minimum.min()
        for i in range(len(upCrossDf)) :
            ax.plot( [upCrossDf.upCrossTime[i], upCrossDf.upCrossTime[i]] , [min_, max_]  , "--" , label = None )
    else :
        ax.plot( upCrossDf.upCrossTime , [0. for i in range(len(upCrossDf))]  , "+" , label = "Cycle bounds" )
    ax.plot( upCrossDf.upCrossTime.iloc[-1] + upCrossDf.Period.iloc[-1] , 0.  , "+" , label = None)
    ax.plot( upCrossDf.MinimumTime , upCrossDf.Minimum , "o" , label = "Min", color ="r")
    ax.plot( upCrossDf.MaximumTime , upCrossDf.Maximum , "o" , label = "Max", color ="b")
    ax.legend(loc = 2)
    return ax


