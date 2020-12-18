"""
   Handle standard interpolation
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

"""

"""

def InterpolateSpline(xx, yy, kind='linear', n=0, ext='extrapolate',fill_value=None):
    """ Get interpolation spline with linear or log interpolation
    """
    
    #Use fill_value outside bounds
    if (ext=='const') and (fill_value is not None):
        if not hasattr(fill_value,'__len__'): fill_value = [fill_value,fill_value]
        if len(fill_value)!=2: raise ValueError('"fill_value" should have 2 values')
        eps = 1e-12
        xx = np.array(xx)
        np.insert(xx,0,xx[0]-eps)
        np.append(xx,[xx[-1]+eps])
        yy=np.array(yy)
        np.insert(yy,0,fill_value[0])
        np.append(yy,[fill_value[1]])
    
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


