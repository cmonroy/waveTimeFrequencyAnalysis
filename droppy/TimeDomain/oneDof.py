import numpy as np
from scipy.integrate import odeint, solve_ivp
from math import pi
import pandas as pd

class OneDof(object):
    """
    Class to solve simple 1 DOF mechanical equation
    M d2x/dt2 + Bl dx/dt + Bq (dx/dt)*|(dx/dt)| + K x = F
    """

    def __init__(self , m,  k , bl , bq ):
        self.m = m
        self.bl = bl
        self.bq = bq
        self.k = k

        self.T0 = 2 * pi * ( self.m / self.k )**0.5

    def __str__(self):
        """
           Print the system parameters
        """

        Bcr = 2*(self.m * self.k)**0.5

        str_ = """!--- Mechanical system parameter ---
Mass : {:.2f}
Stiffness : {:.2f}
Natural period : {:.2f}
Critical damping : {:.2f}
Linear damping : {:.2f} ( = {:.2f} Bcr)
!-----------------------------------!""".format(self.m , self.k, self.T0 , Bcr, self.bl, self.bl/Bcr )

        return str_

    def deriv(self, t , y, f_ex = lambda x, y: 0):
        return np.array( [ y[1] , (-self.bl*y[1] - self.bq * y[1] * abs(y[1])  - self.k*y[0] + f_ex(t,y) ) / self.m ]  )


    def decay(self , tMin , tMax , X0 , t_eval = None ):
        """
          Simulate a decay test (no excitation)
        """

        if type(t_eval) == int :
            t_eval = np.arange(tMin, tMax, self.T0 / t_eval  )

        out = solve_ivp( fun = self.deriv, t_span = [tMin, tMax], y0 = X0, t_eval = t_eval)
        return pd.Series(  index = out.t , data = out.y[0,:]  )

    def forcedMotion(self , tMin, tMax, X0 , f_ex , t_eval) :
        if type(t_eval) == int :
            t_eval = np.arange(tMin, tMax, self.T0 / t_eval  )

        out = solve_ivp( fun = lambda t,y : self.deriv(t,y,f_ex), t_span = [tMin, tMax], y0 = X0, t_eval = t_eval)
        return pd.Series(  index = out.t , data = out.y[0,:]  )

if __name__ == "__main__" :

    m = 15
    bl = 1.5
    bq = 2.0
    k = 10

    # Generate a mcnSolve test
    oneDof = OneDof(m=m, bl=bl, bq=bq, k=k)
    print (oneDof)
    res = oneDof.decay(tMin=0.0, tMax=100.0, X0=np.array([10.0, 0.]) , t_eval = np.arange(0,100,0.1))
    res.plot()

    res2 = oneDof.forcedMotion(tMin=0.0, tMax=100.0, X0=np.array([10.0, 0.]) , t_eval = np.arange(0,100,0.1), f_ex = lambda x,y : x)
    res2.plot()
