# -*- coding:Latin-1 -*-
#!/usr/bin/python

"""
     Slamming signal post-processing

     Auteur : A. Benhamou
     Date : 30/11/18
"""

import os
import pandas as pd
import numpy as np

def analyzeSignal(df,time=None,col=None):
    """ Compute and return analysis data of slamming impact force
        peakValue: value of maximum force
        tPeak: time instant where max force is reached
        tR: rise time
        tD: decrease time
        m0: moment of force (area below curve)
    """

    if type(df)==pd.core.frame.DataFrame: ff = df[col]
    elif type(df)==pd.core.series.Series: ff = df
    elif time is not None: ff = pd.Series(index=time,data=df)
    else: raise(Exception('ERROR: Please provide pandas series or use time or col options'))
    
    res = dict()  
    res['peakValue']  = ff.max()
    res['tPeak'] = ff.idxmax()
    
    #searching for tR
    t0 = res['tPeak']
    for row in ff[res['tPeak']::-1].iteritems():
        t0 = row[0]
        if row[1]<=0.0: break
    res['tR'] = res['tPeak'] - t0
    res['t0'] = t0

    # searching for tD
    # t2 is taken as the minimum within zero-crossing and triangle extrapolation
    for row in ff[res['tPeak']:].iteritems():
        if row[1]>0.5*res['peakValue']: continue
        else:
            t1 = row[0]
            break
    try: t1
    except: t1 = ff.index.max()
    
    ta = 2.0*t1-res['tPeak']
    tb = res['tPeak']
    for row in ff[res['tPeak']:].iteritems():
        tb = row[0]
        if row[1]<=0.0: break
    t2 = min(ta,tb)
    res['tD'] = (t2 - res['tPeak'])
    res['t2'] = t2

    # searching for m0
    # taken as integral between t0 and t2
    res['m0'] = np.trapz(ff[t0:t2], x=ff[t0:t2].index)
    # res['m0'] = 0.5*(res.tR+res.tD)*res['peakValue']
    
    return res