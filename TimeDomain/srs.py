"""
from https://github.com/dsholes/python-srs MIT license 
very slightly modified
"""

from scipy.signal import lfilter
from scipy import integrate
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# Constants
G_TO_MPS2 = 9.81

# Plot formatting
COLORS = sns.color_palette().as_hex()
AX_LABEL_FONT_DICT = {'size':14}
AX_TITLE_FONT_DICT = {'size':16}



class ShockResponseSpectrum:
    def __init__(self, input_se):

        self.input_time_s = input_se.index
        self.input_accel_g = input_se.values

        self.input_accel_mps2 = self.input_accel_g * G_TO_MPS2 # convert accel to m/s^2 for integration to velocity (m/s)
        self.input_vel_mps = integrate.cumtrapz(self.input_accel_mps2, self.input_time_s,initial=0.)

    def run_srs_analysis(self, fn_array, damp = 0.05):
        """
        Review Smallwood method in his paper:
            - 'AN IMPROVED RECURSIVE FORMULA FOR CALCULATING SHOCK RESPONSE SPECTRA'
            - http://www.vibrationdata.com/ramp_invariant/DS_SRS1.pdf

        fn_array in Hz
        """

        self.damp = damp
        self.fn_array = fn_array

        # Should I give user access to the following coefficients??
        T = np.diff(self.input_time_s).mean() # sample interval
        omega_n = 2. * np.pi * self.fn_array
        omega_d = omega_n * np.sqrt(1 - damp**2.)
        E = np.exp(-damp * omega_n * T)
        K = T*omega_d
        C = E*np.cos(K)
        S = E*np.sin(K)
        S_prime = S/K
        b0 = 1. - S_prime
        b1 = 2. * (S_prime - C)
        b2 = E**2. - S_prime
        a0 = np.ones_like(self.fn_array) # Necessary because of how scipy.signal.lfilter() is structured
        a1 = -2. * C
        a2 = E**2.
        b = np.array([b0,b1,b2]).T
        a = np.array([a0,a1,a2]).T

        # Calculate SRS using Smallwood ramp invariant method
        self.pos_accel = np.zeros_like(self.fn_array)
        self.neg_accel = np.zeros_like(self.fn_array)

        adim = np.abs( self.input_accel_g).max()

        for i,f_n in enumerate(self.fn_array):
            output_accel_g = lfilter(b[i], a[i], self.input_accel_g)
            self.pos_accel[i] = output_accel_g.max() / adim
            self.neg_accel[i] = np.abs(output_accel_g.min()) / adim

        return pd.DataFrame( index = fn_array, data = {"POS" : self.pos_accel, "NEG" : self.neg_accel } )


    def run_srs_analysis_ode( self, fn_array, damp = 0.05 ):
        """Compute Shock Response Spectrum
        Raw implementation, solving in time domain for each w0
        """
        from droppy.TimeDomain.oneDof import OneDof
        from scipy.interpolate import InterpolatedUnivariateSpline
        k = self.input_accel_g.max()
        res = pd.DataFrame( index = fn_array, columns = ["POS", "NEG"] )

        # Calculate SRS using Smallwood ramp invariant method
        self.pos_accel = np.zeros_like(self.fn_array)
        self.neg_accel = np.zeros_like(self.fn_array)
        for i, f in enumerate(fn_array) :
            w0 = f*2*np.pi
            m = k  / w0**2
            oneDof = OneDof( m = m, k = k, bl = damp * 2*(m*k)**0.5, bq = 0 )
            T = 2*np.pi*(m/k)**0.5
            interp = InterpolatedUnivariateSpline(self.input_time_s, -self.input_accel_g, ext = 1)
            tMax = max( self.input_time_s.max() , 2 * T)
            resp = oneDof.forcedMotion( tMin = 0.0, tMax=tMax , X0 = [0,0], f_ex= interp, t_eval = np.arange(0,tMax, T/20) )
            self.pos_accel[i] = resp.max()
            self.neg_accel[i] = np.abs(resp.min())
        return pd.DataFrame( index = fn_array, data = {"POS" : self.pos_accel, "NEG" : self.neg_accel } )


    def export_srs_to_csv(self, filename):
        data_array = np.array([self.fn_array,self.pos_accel,self.neg_accel]).T
        cols = ['Natural Frequency (Hz)', 'Peak Positive Accel (G)', 'Peak Negative Accel (G)']
        srs_output_df = pd.DataFrame(data = data_array,
                                     columns = cols)
        srs_output_df.to_csv(filename,index=False)

    def _make_accel_subplot(self,ax = None):

        if ax is not None  :
            fig , ax = plt.subplots()
        ax.plot(self.input_time_s, self.input_accel_g,
                label = 'Accel',
                color = COLORS[0],
                linestyle = '-')
        # leg = ax.legend(fancybox=True,framealpha=1,frameon=True)
        # leg.get_frame().set_edgecolor('k')
        ax.grid(True, which = "both")
        ax.set_xlabel('Time (sec)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_ylabel('Accel (G)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_title('Base Input',
                      fontdict = AX_TITLE_FONT_DICT)
        return ax

    def _make_vel_subplot(self,ax=None):
        if ax is not None  :
            fig , ax = plt.subplots()
        ax.plot(self.input_time_s, self.input_vel_mps,
                label='Vel',
                color=COLORS[0],
                linestyle='-')
        # leg = ax.legend(fancybox=True,framealpha=1,frameon=True)
        # leg.get_frame().set_edgecolor('k')
        ax.grid(True, which="both")
        ax.set_xlabel('Time (sec)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_ylabel('Velocity (m/s)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_title('Base Input',
                      fontdict=AX_TITLE_FONT_DICT)
        return ax

    def _make_srs_subplot(self,requirement, ax = None):
        if ax is not None  :
            fig , ax = plt.subplots()

        ax.loglog(self.fn_array,self.pos_accel,
                  label='Positive',
                  color=COLORS[0],
                  linestyle='-')
        ax.loglog(self.fn_array,self.neg_accel,
                   label='Negative',
                   color=COLORS[0],
                   linestyle='--')
        if requirement is not None:
            self.protocol_fn, self.protocol_accel = requirement
            ax.loglog(self.protocol_fn, self.protocol_accel,
                      color=COLORS[3],
                      linewidth=2,
                      label='Requirement')

        leg = ax.legend(fancybox=True,
                        framealpha=1,
                        frameon=True)
        leg.get_frame().set_edgecolor('k')
        ax.grid(True, which="both")
        ax.set_xlabel('Natural Frequency (Hz)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_ylabel('Peak Accel (G)', fontdict = AX_LABEL_FONT_DICT)
        ax.set_title('Acceleration Shock Response Spectrum (B={0:.1f})'.format(self.damp),
                      fontdict=AX_TITLE_FONT_DICT)
        return ax

    def plot_results(self, requirement = None, filename = None):
        fig = plt.figure()

        gs = gridspec.GridSpec(3,4)
        gs.update(hspace=0.5,wspace=0.75)

        # Create Axes for Input Acceleration and Velocity
        ax0 = fig.add_subplot(gs[0, 0:2])
        ax1 = fig.add_subplot(gs[0, 2:])

        # Create Axis for SRS Output
        ax2 = fig.add_subplot(gs[1:,0:])


        ax0 = self._make_accel_subplot(ax0)
        ax1 = self._make_vel_subplot(ax1)
        ax2 = self._make_srs_subplot(ax2, requirement)

        fig.set_size_inches(10,10)
        if filename is not None:
            fig.savefig(filename,dpi=200)

    def plot_input_accel(self, ax = None):
        if ax is None :
            fig, ax = plt.subplots()
        ax = self._make_accel_subplot(ax)
        fig.set_size_inches(10,7)
        return ax

    def plot_input_vel(self, filename = None):
        if ax is None :
            fig, ax = plt.subplots()
        ax = self._make_vel_subplot(ax)
        return ax


    def plot_srs(self, requirement = None, filename = None):
        fig, ax = plt.subplots()
        ax = self._make_srs_subplot(ax, requirement)
        fig.set_size_inches(10,7)
        if filename is not None:
            fig.savefig(filename,dpi=200)