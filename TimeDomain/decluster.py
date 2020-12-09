import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from droppy.TimeDomain import UpCrossAnalysis
from scipy.optimize import brentq
from droppy import logger

class Decluster( object ) :

    def __init__(self, se , threshold, method, minSpacing, jitted = False ):
        """
        Parameters
        ----------
        se : pd.Series
            Time series to decluster. Index is time.
        threshold : float
            Threshold
        method : str, optional
            Way to decluster. Among [ "updown, "upcross" ].
        minSpacing : str, se.index.dtype.type
            Minimum spacing between maxima. The default is None.
        jitted : bool
            Use Numba and Just-In-Time compilation (beneficial for very large case)

        """

        logger.debug(f"Declustering with threshold = {threshold:}")
        self.se = se
        self.threshold = threshold
        self.method = method
        self.minSpacing = minSpacing
        self.jitted = jitted
        self._declustered = None

        self._do_declustering()



    @classmethod
    def From_Threshold_RP( cls, se, RP, **kwargs ):
        """
        Parameters
        ----------
        se : pd.Seris
            Time series to decluster. Index is time.
        N : float
            Return period of the theshold
        method : str, optional
            Way to decluster. The default is "upcross".
        minSpacing : str, se.index.dtype.type
            Minimum spacing between maxima. The default is None.
        threshold_min : float, optional
            lower bound for solve algorithm, default to se.mean()
        threshold_max : float, optional
            Upper bound for solve algorithm, default to se.mean()
        large_fast : bool, or tuple, optional
            For large time-series, first find boundaries without applying the minspacing criteria (slow for low treshold), default to False
        xtol : int
            xtol tolerance passed to brenth algorythm
        rtol : float
            Relative tolerance

        """
        N =  (se.index[-1] - se.index[0] ) / RP
        logger.debug(f"Declustering : Find threshold for RP={str(RP):} , corresponding to {N:} exceedance")
        return cls.From_Threshold_N( se, N, **kwargs)




    @classmethod
    def From_Threshold_N( cls, se, N, *, threshold_min = None , threshold_max = None , large_fast = False,
                          rtol = 0.0001, xtol = 1e-3, **kwargs ):
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
        threshold_min : float, optional
            lower bound for solve algorithm, default to se.mean()
        threshold_max : float, optional
            Upper bound for solve algorithm, default to se.mean()
        large_fast : bool, or tuple, optional
            For large time-series, first find boundaries without applying the minspacing criteria (slow for low treshold), default to False
        xtol : int
            Absolute tolerance
        rtol : float
            Relative tolerance
        """

        if threshold_min is None:
            threshold_min = se.mean()
        if threshold_max is None:
            threshold_max = se.max()

        minSpacing = kwargs.pop("minSpacing")

        def target( x , minSpac ) :
            peaks = Decluster( se, x, minSpacing = minSpac, **kwargs )
            res = len(peaks.declustered) - N
            logger.debug(f"Declustering solve Threshold = {x} , N = {res+N:}  ")
            return res

        if large_fast :
            if isinstance( large_fast , bool ) :
                large_fast = (0.7 , 1.0)
            logger.debug(f"Starting threshold solve, between {threshold_min:} and {threshold_max:}, without minSpacing")
            threshold_ = brentq( target, args = ( None ), a = threshold_min , b = threshold_max , xtol = xtol, rtol = rtol  )
            threshold_min = threshold_ * large_fast[0]
            threshold_max = threshold_ * large_fast[1]

        logger.debug(f"Starting threshold solve, between {threshold_min:} and {threshold_max:}")
        threshold = brentq( target, args = ( minSpacing ), a = threshold_min , b = threshold_max , xtol = xtol, rtol = rtol  )
        return cls( se=se , threshold = threshold , minSpacing = minSpacing, **kwargs)


    @property
    def exceedance(self):
        return self.declustered - self.threshold

    @property
    def declustered(self):
        if self._declustered is None :
            self._do_declustering()
        return self._declustered

    @property
    def n_exceedance(self):
        #Number of cluster
        logger.warning( "Use n_c instead of n_exceedance now" )
        return len (self.declustered)

    @property
    def n_c(self):
        #Number of cluster
        return len(self.declustered)



    def n_c_u(self , unit):
        #Return number of cluster per unit of time
        return self.n_c * unit /  self.duration()


    @property
    def n_u(self) :
        #Numbez of event above threshold (all event from each cluster are accouted)
        return np.sum( self.se > self.threshold )



    def _do_declustering(self ) :
        """ Perform the actual declsutering


        Returns
        -------
        None.

        """
        #Decluster data
        if "upcross" in self.method.lower():
            peaks = UpCrossAnalysis.FromTs( self.se, upCrossID = None , threshold = self.threshold, method = "upcross")
            self._declustered = pd.Series( index = peaks.MaximumTime , data =  peaks.Maximum.values )

        elif self.method.lower() == "updown" :
            peaks = UpCrossAnalysis.FromTs( self.se, self.threshold, method = "updown" )
            self._declustered = pd.Series( index = peaks.MaximumTime , data =  peaks.Maximum.values )

        elif self.method.lower() == "no" :  # No declustering
            self._declustered = self.se.loc[ self.se > self.threshold ]

        else :
            raise(Exception( "Decluster type not handled" ))

        if self.minSpacing is not None :
            self._declustered = minSpacingFilter(self._declustered , spacing = self.minSpacing, jitted = self.jitted)


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


    def getIntervals(self):
        """Get time to threshold exceedance

        Returns
        -------
        np.ndarray
            Time to failure (interval + 1st time of first max).

        """
        return np.insert(  np.diff( self.exceedance.index ) , 0 , self.exceedance.index[0] - self.se.index[0] )


def minSpacingFilter(se , spacing, jitted = False) :
    """ Remove maxima with small spacing

    Note : room for optimization.

    Parameters
    ----------
    se : pd.Series
        Maxima
    spacing : se.index.dtype
        Minimum interval
    jitted : bool
        Use Numba and Just-In-Time compilation (beneficial for very large case)

    Returns
    -------
    pd.Series
        Maxima with at least "spacing" spacing.
    """

    logger.debug(f"Min spacing filter, {str(spacing):}")

    t = se.index.values
    v = se.values

    if jitted :
        import numba as nb
        minSpacingFilter_array_jitted = nb.jit( nb.types.Tuple( (nb.typeof(t) ,nb.typeof(v)) )
                                               (nb.typeof(t), nb.typeof(v), nb.float64),
                                               nopython = True )(minSpacingFilter_array)
        t_new , v_new = minSpacingFilter_array_jitted( time = t , value = v , spacing = spacing )
    else :
        t_new , v_new = minSpacingFilter_array( time = se.index.values , value = se.values , spacing = spacing )
    return pd.Series( index = t_new , data = v_new )


def minSpacingFilter_array( time , value , spacing) :
    """ Remove maxima with small spacing

    Note : room for optimization.

    Parameters
    ----------
    time : array-like
        Time
    time : array-like
        Values
    spacing : se.index.dtype
        Minimum interval

    Returns
    -------
    (time[:] , value[:])
        Declustered time series
    """

    diff = time[1:] - time[:-1]

    duplicates = np.where(diff < spacing)[0]
    toRemoveList = []

    for dup in duplicates :
        toRemove = dup + np.argmax( np.array([ value[ dup + 1 ] , value[ dup ]]) )
        if toRemove not in toRemoveList and toRemove + 1 not in toRemoveList and toRemove - 1 not in toRemoveList :
            toRemoveList.append( toRemove )

    time_dec = np.delete( time , toRemoveList )
    value_dec = np.delete( value , toRemoveList )

    if len(toRemoveList) > 0 :
        return minSpacingFilter_array(time_dec , value_dec ,  spacing)

    return time_dec , value_dec





if __name__ == "__main__" :


    print("Run")

    time = np.arange(0, 100, 0.5)
    se = pd.Series( index = time , data = np.cos(time) )
    dec = Decluster( se, threshold = 0.2, minSpacing = 10, method = "updown")
    # test = peaksMax( se, 0.5)
    dec.plot()



