"""
   Handle standard interpolation
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

"""

"""

def InterpolateSpline(xx, yy, kind='linear', n=0, ext='extrapolate'):
    """ Get interpolation spline with linear or log interpolation
    """
    
    if kind=='linear':
        spline = InterpolatedUnivariateSpline( xx , yy, k=1, ext=ext)
        if n==0: return spline
        elif n==1: return spline.derivative(n=1)
    elif kind=='log':
        ispos = (xx>0.0) & (yy>0.0)
        logx = np.log10(xx[ispos])
        logy = np.log10(yy[ispos])
        lin_spline = InterpolatedUnivariateSpline(logx, logy, k=1, ext=ext)
        
        if n==0:
            log_spline = lambda zz: np.power(10.0, lin_spline(np.log10(zz)))
            return log_spline
        elif n==1: 
            lin_dspline = lin_spline.derivative(n=1)
            log_dspline = lambda zz: lin_dspline(np.log10(zz))/zz*np.power(10.0, lin_spline(np.log10(zz)))
            return log_dspline


