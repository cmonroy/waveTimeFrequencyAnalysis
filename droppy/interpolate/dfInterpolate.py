
"""

Interpolate dataframe
(do not know why it does not exists within pandas)

"""
from __future__ import print_function , division
from scipy.interpolate import interp1d , InterpolatedUnivariateSpline  #InterpolatedUnivariateSpline is much much faster
import pandas as pd
import numpy as np
from droppy.interpolate import InterpolateSpline


#To make the interpolator able to linearly interpolate
def extrap1d(interpolator , kind = "linear" ):
    xs = interpolator.x
    ys = interpolator.y
    if kind == "linear" :
       def pointwise(x):
           if x < xs[0]:     return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
           elif x > xs[-1]:  return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
           else:             return interpolator(x)
    elif kind == "constant" :
       def pointwise(x):
           if x < xs[0]:     return ys[0]
           elif x > xs[-1]:  return ys[-1]
           else:             return interpolator(x)
    else :
       print (kind , "extrapolator not supported")
       exit()
    def ufunclike(xs):
        return np.array(map(pointwise, np.array(xs)))
    return ufunclike


def interpolate( df , newIndex=None,  newColumns=None, xfunc = lambda x : x , yfunc = (lambda y : y , lambda y : y), interpolator = InterpolatedUnivariateSpline   , **kwargs) :
    """ Interpolate dataFrame with new index
    
    :param df: DataFrame to interpolate
    :param newIndex: Index of the interpolated dataFrame
    :param xfunc: Tuple of function and inverse function to apply to index before and after interpolation (for a log interpolation for instance)
    :param yfunc: Tuple of function and inverse function to apply to values before and after interpolation (for a log interpolation for instance)

    """
    ndf = df.copy()
    
    if newIndex is not None:
        newData = {}
        for col in ndf.columns :
            f = interpolator( xfunc( ndf.index ) , yfunc[0]( ndf[col] )  , **kwargs )
            newData[col] =  yfunc[1]( f( xfunc(newIndex ) ) )
        ndf = pd.DataFrame(index = newIndex , data = newData)
   
    if newColumns is not None:
        newData = {}
        for idx in ndf.columns :
            f = interpolator( xfunc( ndf.columns ) , yfunc[0]( ndf[idx] )  , **kwargs )
            newData.loc[idx] =  yfunc[1]( f( xfunc(newColumns ) ) )
        ndf = pd.DataFrame(columns = newColumns, data = newData)
    
    return ndf

def interpolateValue( df, newIndex, col=0, k = 3, interpolator = InterpolatedUnivariateSpline,  **kwargs ) :
   if type(df) == pd.core.frame.DataFrame :
      val = interpolator( df.index , df[col] , k = k)(newIndex)
   else :
      val = interpolator( df.index , df , k = k)(newIndex)
   return val

def interpDf( df , newIndex=None,  newColumns=None, kind=['linear','linear'], **kwargs):
    """ Interpolate dataFrame with new index (simple version of interpolate)
    
    :param df: DataFrame to interpolate
    :param newIndex: Index of the interpolated dataFrame
    :param newColumn: Columns of the interpolated dataFrame
    """
    ndf = df.copy()
    
    if newIndex is not None:
        newData = {}
        #Deal with datetime index
        if isinstance(ndf.index,pd.DatetimeIndex): oldIndexF = ndf.index.values.astype(float)
        else: oldIndexF = ndf.index
        if isinstance(newIndex,pd.DatetimeIndex): newIndexF = newIndex.values.astype(float)
        else: newIndexF = newIndex
        for col in ndf.columns :
            f = InterpolateSpline( oldIndexF , ndf[col], kind=kind[0], **kwargs )
            newData[col] =   f( newIndexF )
        ndf = pd.DataFrame(index = newIndex, data = newData)
   
    if newColumns is not None:
        newData = {}
        for idx in ndf.index :
            f = InterpolateSpline( ndf.columns , ndf.loc[idx], kind=kind[1], **kwargs )
            newData[idx] = f( newColumns )
        ndf = pd.DataFrame(index = ndf.index, columns = newColumns, data = np.array(list(newData.values())))
    
    return ndf

def invInterpolate( df , newCol,  xfunc = (lambda y : y , lambda y : y) , yfunc =  lambda y : y , extrapolate = False , **kwargs) :
   """ The col is now the index, with interpolated values of index as columns, xfunc apply to the original index
   WARNING : PROBLEM if the function is not monotonic !!
   """
   newData = {}
   for col in df.columns :
      f = interp1d( yfunc( df[col] ) , xfunc[0]( df.index )  , **kwargs )
      if extrapolate : f = extrap1d(f , extrapolate )
      newData[col] =   xfunc[1](f( yfunc(newCol) ))
   return pd.DataFrame(index = newCol , data = newData)

if __name__ == "__main__" :

   import numpy as np

   from matplotlib import pyplot as plt

   x_old = np.linspace( 1 , 100 , 10)
   x_new = np.linspace( 1 , 100, 100)

   y_new = np.linspace( 0 , 6 , 100)

   df = pd.DataFrame( index = x_old , data = {"cos" : np.log(x_old) } )
   ax = df.plot(marker = "o" , linestyle = "")
   

   dfNew = interpolate( df , x_new , xfunc = np.log, extrapolate = "linear" ,  kind = "linear"  )
   dfNew = interpolate( df , x_new , extrapolate = "linear" ,  kind = "linear"  )
   
   ax = dfNew.plot(ax=ax)
   
   plt.show()
   exit()

   #dfNew2 = invInterpolate( df , y_new , xfunc = (np.log, np.exp) , extrapolate = True,  kind = "linear"  )
   dfNew2 = invInterpolate( df , y_new , yfunc = np.exp , extrapolate = "linear",  kind = "linear"  )
   ax.plot( dfNew2 , dfNew2.index , "o", label = "inv"  )
   
   ax.legend()
   ax.set_xscale("log")
   ax.set_xlim( [0,1000] )

   plt.show()
