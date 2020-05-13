import pandas as pd
import numpy as np
from scipy.stats import rayleigh
from matplotlib import pyplot as plt
from droppy.pyplotTools.statPlots import distPlot




class UpCrossAnalysis( pd.DataFrame ):
    """
    
    Object-oriented interface to upcrossing analysis
    
    """
    
    def __init__(self, *args, **kwargs) : 
        pd.DataFrame.__init__(self, *args, **kwargs)
        self.se = None
        
    @property
    def _constructor(self):
        return UpCrossAnalysis

    @classmethod
    def Merge( cls, listUpCross ):
        """
        Merge several upcrossing analysis

        Parameters
        ----------
        listUpCross : list of upCrossing anlysis
            list of upCrossing anlysis

        Returns
        -------
        UpCrossAnalysis
            Merged data.

        """
        return UpCrossAnalysis( pd.concat( listUpCross ).reset_index() )


    @classmethod
    def FromTs( cls, se, threshold = "mean", method = "upcross"):
        """
        
        Parameters
        ----------
        se : pd.Series
            Time trace to analyse
        threshold : flaot, optional
            upcrossing threshold. The default is "mean".

        Returns
        -------
        res : UpCrossing analysis
            Up-crossing data

        """
        
        if method.lower() == "upcross" : 
            res = cls(upCrossMinMax( se, threshold = threshold ))
        elif method.lower() == "updown" : 
            res = cls(peaksMax( se, threshold = threshold ))
        else : 
            raise(Exception(f"method '{method:}' not available"))

        res.se = se        
        return res
    
    def plot(self , ax = None, **kwargs):
        """Plot time series together with maximum, minimum and cycle bounds
        """
        ax = plotUpCross( self, ax=ax,  **kwargs )
        if self.se is not None :
            self.se.plot(ax=ax)
        return ax
        

    def plotDistribution( self, ax = None, data = "Maximum", addRayleigh = None, **kwargs ):
        """Plot upcrossing distribution
        
        Parameters
        ----------
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        data : TYPE, optional
            DESCRIPTION. The default is "Maximum".
        addRayleigh : None, float or "auto", optional
            Plot Rayleight distribution
            addRayleigh == "auto" : standard deviation calculated from time series
            addRayleigh == float : standard deviation given
            addRayleigh == None : Do not plot
            The default is None.

        Returns
        -------
        ax : TYPE
            DESCRIPTION.

        """
        
        if ax == None : 
            fig, ax = plt.subplots()
            
        
        if addRayleigh is not None : 
            if addRayleigh == "auto":
                addRayleigh = rayleigh(0.0,  scale = self.se.std() )
            else :     
                addRayleigh = rayleigh(0.0,  scale = addRayleigh )
            
        distPlot( data = self.loc[: , data].values,
                  frozenDist = addRayleigh,
                  ax=ax, 
                  **kwargs
                  )
        
        return ax 
    



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

    if len(up_) == 0 : 
        return np.array([], dtype = int) , np.array([], dtype = int)

    if down_[0] < up_[0] :
        down_ = down_[1:]

    if len(down_) != len(up_):
        up_ = up_[:-1]

    return up_, down_


def peaksMax( se, threshold ) : 

     up_, down_ = getPeaksBounds( se, threshold )
     
     maxIndex = np.empty( up_.shape , dtype = int )  

     for i in range(len( up_ )) : 
         maxIndex[i] = up_[i] + se.values[ up_[i] : down_[i]+1 ].argmax()
     
     return pd.DataFrame( data = { "Maximum" : se.iloc[ maxIndex  ] , 
                                   "MaximumTime" : se.index[ maxIndex ] ,
                                   "upCrossTime" : se.index[ up_ ] , "downCrossTime":  se.index[ down_ ], 
                                   "Period" : se.index[ down_ ] - se.index[ up_ ]} )
     


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

def getUpCrossDist(upCrossDf) :
    """
       Get Up-crossing distribution from upCrossMinMax result
    """
    
    N = upCrossDf.shape[0]
    p_ex = np.arange(1./N,1.+1./N,1./N)
    df = pd.DataFrame(index=p_ex,columns=['Minimum','Maximum'])
    df.Minimum = upCrossDf.Minimum.sort_values(ascending=True).values
    df.Maximum = upCrossDf.Maximum.sort_values(ascending=False).values
    return df

def plotUpCross( upCrossDf , ax = None, cycleLimits = False ) :
    """
       Plot time trace together with extracted maxima
    """
    from matplotlib import pyplot as plt
    if ax is None : fig , ax = plt.subplots()
    if cycleLimits :
        for i in range(len(upCrossDf)) :
            ax.axvline( x = upCrossDf.upCrossTime[i] , label = None , alpha = 0.3)
            ax.axvline( x = upCrossDf.upCrossTime[i] + upCrossDf.Period[i] , label = None, alpha = 0.3)
    # else :
    #     ax.plot( upCrossDf.upCrossTime , [0. for i in range(len(upCrossDf))]  , "+" , label =  )
    #     ax.plot( upCrossDf.upCrossTime.iloc[-1] + upCrossDf.Period.iloc[-1] , 0.  , "+" , label = None)

    ax.plot( upCrossDf.MaximumTime , upCrossDf.Maximum , "o" , label = "Max", color ="b")
    
    if "MinimumTime" in upCrossDf.columns : 
        ax.plot( upCrossDf.MinimumTime , upCrossDf.Minimum , "o" , label = "Min", color ="r")
    
    ax.legend(loc = 2)
    return ax

def plotUpCrossDist( upCrossDist , ax = None, label=None):
    """
       Plot upcrossing distribution
    """
    from matplotlib import pyplot as plt
    if ax is None : fig , ax = plt.subplots()
    prob = np.concatenate([upCrossDist.index,upCrossDist.index[::-1]])
    values = np.concatenate([upCrossDist.Minimum.values,upCrossDist.Maximum.values[::-1]])
    ax.plot(values,prob,'-+',label=label)
    ax.set_ylabel('Exeedence probability')
    ax.set_yscale('log')
    if label is not None: ax.legend()
    return ax




    
    
    
    
    
    
    
    
    
    
    
        
        
        