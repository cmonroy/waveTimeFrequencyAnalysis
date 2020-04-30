import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from droppy.TimeDomain import upCrossMinMax, peaksMax

class Decluster( object ) :
    
    def __init__(self, se , threshold, method = "upcross", minSpacing = None ):
        """
        Parameters
        ----------
        se : pd.Seris
            Time series to decluster. Index is time. 
        threshold : float
            Threshold
        method : str, optional
            Way to decluster. The default is "upcross".
        minSpacing : str, se.index.dtype.type
            Minimum spacing between maxima. The default is None.

               
        """

        self.se = se
        self.threshold = threshold
        self.method = method
        self.minSpacing = minSpacing
        self._declustered = None

        
    @classmethod
    def From_Threshold_N( cls, se, N, **kwargs ):
        """
        
        Parameters
        ----------
        se : pd.Seris
            Time series to decluster. Index is time. 
        N : Integer
            Number of exeedance to consider
        method : str, optional
            Way to decluster. The default is "upcross".
        minSpacing : str, se.index.dtype.type
            Minimum spacing between maxima. The default is None.

        """
        
        def target( x ) :
            peaks = Decluster( se, x, **kwargs )
            res = len(peaks.declustered) - N
            return res

        from scipy.optimize import brentq
        threshold = brentq( target, se.mean() , se.max()  , xtol = 0.001  )        
        
        return cls( se=se , threshold = threshold , **kwargs)
        

        
    @property   
    def exceedance(self):
        return self.declustered - self.threshold

    @property
    def declustered(self):
        if self._declustered is None :
            self._do_declustering()
        return self._declustered
            
                        
    def _do_declustering(self ) :
        """ Perform the actual declsutering
        

        Returns
        -------
        None.

        """
        #Decluster data
        if "upcross" in self.method.lower():
            peaks = upCrossMinMax( self.se, upCrossID = None , threshold = self.threshold)
            self._declustered = pd.Series( index = peaks.MaximumTime , data =  peaks.Maximum.values )

        elif self.method.lower() == "no" :  # No declustering
            self._declustered = self.se.loc[ self.se > self.threshold ]

        elif self.method.lower() == "updown" : 
            peaks = peaksMax( self.se, self.threshold )
            self._declustered = pd.Series( index = peaks.MaximumTime , data =  peaks.Maximum.values )
            
        else :
            raise(Exception( "Decluster type not handled" ))
            
        if self.minSpacing is not None : 
                self._declustered = minSpacingFilter(self._declustered , spacing = self.minSpacing)

    
    def plot(self, ax=None) :
        """
        

        Parameters
        ----------
        ax : matplotlib ax
            Where to plot the figure. Created is not provided

        Returns
        -------
        ax : matplotlib ax
            ax with duclestering plot.
        """
        if ax is None : 
            fig , ax = plt.subplots()
        ax.plot(self.se.index , self.se.values)
        ax.plot( self.declustered.index, self.declustered.values , marker = "o" , linestyle = "" )
        ax.hlines( self.threshold, xmin = self.se.index.min() , xmax = self.se.index.max() )
        return ax
    
    
    def duration(self):
        return self.se.index[-1] - self.se.index[0]
    

def minSpacingFilter(se , spacing) :
    """ Remove maxima with small spacing
    

    Parameters
    ----------
    se : pd.Series
        Maxima
    spacing : se.index.dtype
        Minimum interval 

    Returns
    -------
    pd.Series
        Maxima with at least "spacing" spacing.

    """
    diff = se.index[ 1:] - se.index[ :-1 ]
    duplicates = np.where(diff < spacing)[0]
    toRemoveList = []
    
    for dup in duplicates :
        toRemove = dup + np.argmax( [ se.iloc[ dup + 1 ] , se.iloc[ dup ]] )
        if toRemove not in toRemoveList and toRemove + 1 not in toRemoveList and toRemove - 1 not in toRemoveList :
            toRemoveList.append( toRemove )

    se_dec = se.drop( se.index[ toRemoveList ] )

    if len(toRemoveList) > 0 :
        return minSpacingFilter(se_dec , spacing)

    return se_dec




if __name__ == "__main__" : 
    
    
    print("Run")
    
    time = np.arange(0, 100, 0.5)
    se = pd.Series( index = time , data = np.cos(time) )
    dec = Decluster( se, threshold = 0.5, minSpacing = 20, method = "updown")
    test = peaksMax( se, 0.5)
    dec.plot()
    
    