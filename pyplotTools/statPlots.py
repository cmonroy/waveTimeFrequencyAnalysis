import numpy as np
from matplotlib import pyplot as plt


def probN( n ):
   """
     return probability vector
   """
   return 1 - ( np.arange( 1, n+1 ) / (n+1) )


def probN_ci( n, alpha = 0.05, method = "jeff" ):
    """Compute confidence interval for the empirical distribution.

    Statmodel could also be used directly :

    from statsmodels.stats.proportion import proportion_confint
    ci_l[i], ci_u[i] = proportion_confint( p_*n , n , method = method)
    """
    m = np.arange( n, 0, -1 )
    ci_u = np.empty( (n) )
    ci_l = np.empty( (n) )
    if method[:4] == 'jeff':
        for i in range(n):
            ci_l[i], ci_u[i] = beta( m[i]  + 0.5 , 0.5 + n - m[i] ).ppf( [alpha/2 , 1-alpha/2] )  # Jeffrey
    elif method[:4] == 'beta':
        for i in range(n):
            #Clopper-Pierson
            ci_l[i] = beta( m[-i] , 1 + n - m[-i] ).ppf( alpha/2 )
            ci_u[i] = beta( m[-i] + 1 , n - m[-i] ).ppf( 1-alpha/2 )

    return ci_l, ci_u

def qqplot(data, dist, x_y = True, ax=None, **kwargs):
    """
    Standard qqpLot
    """
    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)
    fig, ax = plt.subplots()
    ax.plot( dist.ppf( np.arange(1,n+1)/(n+1) ), np.sort(data), "+" )

    if x_y is True:
        ax.plot( [np.min(data), np.max(data)], [np.min(data), np.max(data)], "-" )

    ax.set_xlabel("Theoretical")
    ax.set_ylabel("Emprical")
    ax.set_title("QQ plot")
    return ax


def qqplot2(data_1, data_2, label_1=None, label_2=None, ax=None, x_y = True, marker = "+", **kwargs):
    """
    QQ plot for two set of data (scatter plot of sorted values)
    """
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot( np.sort(data_1), np.sort(data_2), marker = marker,  linestyle = "", **kwargs )

    if x_y is True :
        ax.plot( [np.min(data_1), np.max(data_1)], [np.min(data_1), np.max(data_1)], "-" )

    if label_1 is not None :
        ax.set_xlabel(label_1)

    if label_2 is not None :
        ax.set_ylabel(label_2)

    ax.set_title("QQ plot")
    return ax



def distPlot(data, frozenDist=None, ax=None, label=None, labelFit=None, marker="+", noData = False, order = 1, alpha_ci = None, period=None, **kwargs ):
    """
    Plot parametric distribution together with data
    """

    from matplotlib import pyplot as plt
    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)
    prob = probN(n)

    if period is not None:
        conv = 3600. / period  #Convert from probability to events rate
    else :
        conv = 1.0

    if not noData:
        if alpha_ci is not None :
            ci_l , ci_u = probN_ci( n , alpha = alpha_ci )
            ax.fill_between( np.sort(data)[::order], ci_l*conv , ci_u*conv , alpha = 0.2, label = label + " C.I." , **kwargs)
        ax.plot(np.sort(data)[::order], prob*conv, marker=marker, label=label, linestyle="", **kwargs)

    if frozenDist is not None:
        ax.plot(frozenDist.isf(prob), prob*conv, "-", label = labelFit, **kwargs)
    ax.set_yscale("log")

    if period is None :
        ax.set_ylabel("Exceedance probability" )
    else :
        ax.set_ylabel("Events rate (n / hour)" )
    if label is not None or labelFit is not None :
        ax.legend()
    ax.grid(True)
    return ax
