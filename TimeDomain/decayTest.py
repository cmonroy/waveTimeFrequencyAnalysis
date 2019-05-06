from __future__ import division
import numpy as np
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from math import pi, log
import pandas as pd


class DecayAnalysis(object):

    def __init__(self, se, method="maxmoy", w_filter = None  ):

        self.se = se
        # Read the mcnSolve test time serie
        self.time = se.index
        self.motion = se.values

        #Method
        self.method = method

        #To be filled
        self.coef = None
        self.maxVal = None
        self.minVal = None

        if w_filter is not None :
            self._smooth( w_filter )


    def _smooth(self, w_filter):
        from droppy.TimeDomain import bandPass
        self.se = bandPass( self.se, fmin=0, fmax = w_filter, unit ="rad/s" )
        self.motion = self.se.values

    def _getExtrema(self):
        """
        Get local extrema. require a smooth signal.
        """
        from droppy.TimeDomain import upCrossMinMax
        upCross = upCrossMinMax( self.se )

        self.maxTime = upCross.MaximumTime
        self.maxVal = upCross.Maximum
        self.minTime = upCross.MinimumTime
        self.minVal = upCross.Minimum


    def getPeriod(self, n=5):
        # Calculate and return the period, based on extrema location
        if self.maxVal is None:
            self._getExtrema()
        T = 0.
        for i in range(n):
            T += self.maxTime[i+1] - self.maxTime[i]
            T += self.minTime[i+1] - self.minTime[i]
        return T / (2*n)

    def plotTimeTrace(self, ax = None):
        # Plot the analysed signal and the extracted maxima
        if ax is None :
            fig, ax = plt.subplots()
        ax.plot(self.time, self.motion, "-")
        if self.maxTime is not None:
            ax.plot(self.maxTime[:], self.maxVal[:], "o")
            ax.plot(self.minTime[:], self.minVal[:], "o")
        return ax

    def _regression(self):
        """
        Perform the linear regression to get p and q
        Beq_adim = delta  / ( 4pi**2 + delta**2 )**0.5
        """
        if self.coef is not None :
            return

        if self.maxVal is None:
            self._getExtrema()

        if self.method == "max":  # Max only
            self.n = np.zeros((2, len(self.maxVal)-1))
            for i in range(len(self.n[0, :])):
                self.n[0, i] = (self.maxVal[i+1] + self.maxVal[i]) * 0.5
                delta = -log(self.maxVal[i+1] / self.maxVal[i])
                self.n[1, i] = delta / (4*pi**2 + delta**2)**0.5

        elif self.method == "min":  # Min only
            self.n = np.zeros((2, len(self.minVal)-1))
            for i in range(len(self.n[0, :])):
                self.n[0, i] = -(self.minVal[i+1] + self.minVal[i]) * 0.5
                delta = -log(self.minVal[i+1] / self.minVal[i])
                self.n[1, i] = delta / (4*pi**2 + delta**2)**0.5

        elif self.method == "maxmoy":  # Max to get the decreement, min to get the amplitude
            if self.minTime[0] < self.maxTime[1]:  # Start from x > 0
                self.n = np.zeros((2, len(self.maxVal)-1))
                for i in range(len(self.n[0, :])):
                    self.n[0, i] = -(self.minVal[i+1])
                    delta = -log(self.maxVal[i+1] / self.maxVal[i])
                    self.n[1, i] = delta / (4*pi**2 + delta**2)**0.5
            else:
                exit("Not node yet")

        elif self.method == "minmax":  # Use semi cycle (min and max)
            nDemi = min(len(self.minVal), len(self.maxVal))
            self.n = np.zeros((2, nDemi))
            for i in range(len(self.n[0, :])):
                self.n[0, i] = (self.maxVal[i] - self.minVal[i]) * 0.5
                delta = -log(self.maxVal[i] / (-self.minVal[i])) * 2.0
                self.n[1, i] = delta / (4*pi**2 + delta**2)**0.5

        A = np.vstack([self.n[0, :], np.ones(len(self.n[0, :]))]).T
        self.coef = np.linalg.lstsq(A, self.n[1, :] , rcond=None)[0]  # obtaining the parameters
        return self.coef


    def plotRegression(self, ax = None, label = ""):
        # Plot the regression
        if ax is None:
            fig, ax = plt.subplots()

        self._regression()
        ax.plot( self.n[0, :], self.n[1, :], 'o',  label="Exp{}".format(label))
        xi = np.linspace(np.min(self.n[0, :]), np.max(self.n[0, :]), 3)
        line = self.coef[0]*xi + self.coef[1]  # regression line
        ax.plot( xi , line, 'r-', label='Regression{}'.format(label))
        ax.legend()
        ax.set_title("Decay regression, method = {}".format(self.method))
        ax.set_xlabel("Amplitude")
        ax.set_ylabel("Equivalent damping (% of critical)")
        ax.text( 0.55, 0.25,  "y = {:.2e} * x + {:.2e}".format( self.coef[0] , self.coef[1]) , transform=ax.transAxes  )

        return ax

    def getDimDampingCoef(self, T0, k, *args , **kwargs):
        """
        Return dimensional damping coefficients
        T0 = Resonace period
        k  = Stiffness
        """
        if self.coef is None :
            self._regression(*args, **kwargs)

        Bcr = T0 * k / pi
        w0 = 2.*pi / T0
        blin = self.coef[1] * Bcr
        bquad = self.coef[0] * Bcr * 3. * pi / (8*w0)
        return blin, bquad



def test():
    from droppy.TimeDomain.oneDof import OneDof
    m = 15
    bl = 1.5
    bq = 2.0
    k = 10

    # Generate a mcnSolve test
    mcnSolve = OneDof(m=m, bl=bl, bq=bq, k=k)
    decayMotion = mcnSolve.decay(tMin=0.0, tMax=100.0, X0 = np.array([10.0, 0.]), t_eval = np.arange(0., 100., 0.1) )

    # test2.plotTimeTrace()
    for method in ["maxmoy", "min", "max", "minmax"]:
        test2 = DecayAnalysis( decayMotion, method = method )
        bl_test, bq_test = test2.getDimDampingCoef(T0=2*pi*(m/k)**0.5, k=k)
        print("Error ", method, " : ",  ((bl_test/bl-1)*100, (bq_test/bq-1)*100), "%")
        test2.plotRegression()
    print("test finished")


if __name__ == "__main__":

    test()






