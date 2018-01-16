import matplotlib.pyplot as plt
import numpy as np
from matplotlib import scale as mscale
from matplotlib import transforms as mtransforms
from matplotlib.ticker import (NullLocator,  Locator, AutoLocator, FixedLocator)
from matplotlib.ticker import (NullFormatter, ScalarFormatter, FuncFormatter)
from scipy.stats import norm

"""

  Custom scales (subclassing of matplotlib.scale.ScaleBase)
  
     -phi (inverse normal scale)
     -...

"""



class PhiLocator(AutoLocator):
    """
    Determine the tick locations
    Simply transform the linear ticks (based on AutoLocator) to phi ticks
    """

    def tick_values(self, vmin, vmax) :
        eps = 0.
        return  norm.cdf(AutoLocator.tick_values( self , norm.ppf( max(eps,vmin) ) , norm.ppf(min(1-eps,vmax)) ))


class PhiScale(mscale.ScaleBase):
    """
       Inverse normal scale
    """
    name = 'phi'

    def __init__(self, axis, autoTicks = True, beta = False,**kwargs):
        self.autoTicks = autoTicks
        self.beta = beta

    def get_transform(self):
        return self.CustomTransform()

    def set_default_locators_and_formatters(self, axis):

        if self.beta :
           axis.set_major_locator( PhiLocator() )
           def beta(x,_):
              return  "{:.1f}".format( norm.ppf( x ) )
           axis.set_major_formatter(  FuncFormatter(beta)  )
        else :
           if self.autoTicks:
              axis.set_major_locator( PhiLocator() )
              axis.set_major_formatter( ScalarFormatter() )
           else :
              from matplotlib.ticker import LogFormatterSciNotation #Import here to avoid error at import with older version of matplotlib (<2.0)
              axis.set_major_locator(FixedLocator([1-1e8, 1e-7, 1e-6, 1e-5, 1e-4 , 1e-3 , 1e-2, 1e-1 , 0.5 , 1-1e-1 , 1-1e-2 , 1-1e-3 , 1-1e-4 , 1-1e-5, 1-1e-6 , 1-1e-7 , 1-1e-8]))
              def fmt(x,_):
                 if x >= 0.9 : return "1-" + LogFormatterSciNotation()(1-x,_ )
                 elif x <= 0.1 : return LogFormatterSciNotation()(x,_ )
                 else : return  "{:}".format(x)
              axis.set_major_formatter( FuncFormatter(fmt)  )
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    
    
    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to values between 0 and 1 (excluded).
        """
        
        eps = 1e-15    # This value should rarely if ever end up with a visible effect.
        return (eps if vmin <= 0 else vmin,  1 - eps if vmax >= 1 else vmax)

    class CustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
           masked = np.ma.masked_where( (a < 1.0) | (a > 0.0) , a)
           return norm.ppf( masked  )
           """
           res = np.empty( a.shape )
           ok =  (a<1.) & (a>0.)
           goodId = np.where( ok )
           badId = np.where( ~ok )
           res[ goodId ] = norm.ppf( a[goodId]  )
           res[ badId ] = np.nan
           return res
           """
        def inverted(self):
            return PhiScale.InvertedCustomTransform()

    class InvertedCustomTransform(mtransforms.Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True

        def __init__(self):
            mtransforms.Transform.__init__(self)

        def transform_non_affine(self, a):
           return norm.cdf(a)

        def inverted(self):
            return PhiScale.CustomTransform()

# Now that the Scale class has been defined, it must be registered so
# that ``matplotlib`` can find it.
mscale.register_scale(PhiScale)


#------------------------------------------------
class BetaScale(PhiScale):
    """Inverse normal scale
       Same as PhiScale (phi) with beta = True
    """
    name = 'beta'
    def __init__(self, axis, **kwargs) :
       PhiScale.__init__( self, axis, beta = True)
mscale.register_scale(BetaScale)


#------------------------------------------------
class Phi2Scale(PhiScale):
    """Inverse normal scale
       Same as PhiScale (phi) with autoTicks = False (ticks are fixed, not always nice)
    """
    name = 'phi2'
    def __init__(self, axis, **kwargs) :
       PhiScale.__init__( self, axis, autoTicks = False)

mscale.register_scale(Phi2Scale)


if __name__ == "__main__":
    """
    b = np.linspace(-8,8,50)
    fig1 , ax1= plt.subplots()
    ax1.plot(  b ,  norm.cdf(b)  , "-" )
    ax1.set_yscale( "phi"  )
    ax1.set_ylim([0.0,1.])
    plt.show()
    """

    from matplotlib import pyplot as plt
    fig , ax = plt.subplots()
    ax.plot( [-0.2, 0.1,0.5 , 0.6] , [-0.1, -0.2, 0.5 , 1.1] , "o")
  
    ax.set_yscale("phi2")
    ax.set_ylim( [0.0,1.0] )
    
    

