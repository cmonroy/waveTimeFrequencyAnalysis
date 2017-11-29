"""
   Handle complex interpolation
"""

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

class InterpolatedComplexSpline( object  ) :
   """  Interpolator for complex numbers, works like (and uses) scipy "InterpolatedUnivariateSpline"

    For 'rotating' field, it is a bit better to interpolate module and phases instead of real and imaginary part.

    Example :
        interpolator = InterpolatedComplexSpline(x_original , y_original , cmplxMethod = "mod" )
        interpvalues = interpolator( x_new )
   """

   def __init__(self,  x , y , cmplxMethod  = "reim", **kwargs ):
      """Construct the interpolator

      x : Original
      y : Original data
      cmplxMethod : Way to deal with complex interpolation ("reim" or "mod" )
      **kwargs : argument accepted by scipy.interpolate.InterpolatedUnivariateSpline
      """

      self.cmplxMethod = cmplxMethod

      #Interpolate re and imaginary separately
      if self.cmplxMethod == "reim" :
         self.f_re = InterpolatedUnivariateSpline( x , np.real(y) , **kwargs )
         self.f_im = InterpolatedUnivariateSpline( x , np.imag(y) , **kwargs )

      #Phase from reim interpolation, module interpolated
      elif self.cmplxMethod == "mod" :
         self.f_abs = InterpolatedUnivariateSpline( x , np.abs(y) , **kwargs )
         self.f_re = InterpolatedUnivariateSpline( x , np.real(y) , **kwargs )
         self.f_im = InterpolatedUnivariateSpline( x , np.imag(y) , **kwargs )
      else :
         raise(NotImplementedError)

   def __call__(self, x) :
      """Interpolate at x location
      """

      if self.cmplxMethod == "reim" :
         return self.f_re(x) + 1.j*self.f_im(x)
      elif self.cmplxMethod == "mod" :
         phase = np.angle( self.f_re(x) + 1.j*self.f_im(x) )
         return self.f_abs(x) *  np.exp( 1j*phase )
      else :
         raise(NotImplementedError)


if __name__ == "__main__" :

    """
       Example
    """

    print ("Illustrate interpolation of complex")
    from matplotlib import pyplot as plt
    s = np.linspace( 0 , 2*np.pi , 10 )
    s_new = np.linspace( 0 , 2*np.pi , 200 )

    a = np.cos(s) + 1j*np.sin(s)

    interp = InterpolatedComplexSpline(s , a, k = 1 )( s_new )
    interp2 = InterpolatedComplexSpline(s , a, cmplxMethod = "mod" )( s_new )

    fig , (ax1,ax2, axre) = plt.subplots(3)
    ax1.plot( np.real(a) , np.imag(a) , "+" , label = "Original")
    ax1.plot( np.real(interp) , np.imag(interp) , "-" , label = "Re/Im Interpolation")
    ax1.plot( np.real(interp) , np.imag(interp2) , "-", label = "Mod/Phi Interpolation" )
    ax1.set_xlabel( "Real" )
    ax1.set_ylabel( "Imag" )
    ax1.legend(loc = 1)

    ax2.plot( s , np.abs(a) , "+" )
    ax2.plot( s_new , np.abs(interp) , "-" )
    ax2.plot( s_new , np.abs(interp2) , "-" )
    ax2.set_xlabel( "x" )
    ax2.set_ylabel( "Module" )

    axre.plot( s , np.real(a) , "+" )
    axre.plot( s_new , np.real(interp) , "-" )
    axre.plot( s_new , np.real(interp2) , "-" )
    axre.set_xlabel( "x" )
    axre.set_ylabel( "Real Part" )
    fig.tight_layout()
    plt.show()


