import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta

def probN( n, alphap = 0.0, betap = 0.0):
    """Return exceedance probability of the ranked data
    
    (inverse of scipy.stats.mquantiles)

    
    Parameters
    ----------
    n : int
        Size of the data vector
    alphap : float, optional
        Plotting positions parameter. The default is 0.0.
    betap : float, optional
        Plotting positions parameter. The default is 0.0.

    Returns
    -------
    np.ndarray
        Exceedance probability of the ranked data.



    Typical values of (alphap,betap) are:
        - (0,1)    : ``p(k) = k/n`` : linear interpolation of cdf
          (**R** type 4)
        - (.5,.5)  : ``p(k) = (k - 1/2.)/n`` : piecewise linear function
          (**R** type 5)
        - (0,0)    : ``p(k) = k/(n+1)`` :
          (**R** type 6)
        - (1,1)    : ``p(k) = (k-1)/(n-1)``: p(k) = mode[F(x[k])].
          (**R** type 7, **R** default)
        - (1/3,1/3): ``p(k) = (k-1/3)/(n+1/3)``: Then p(k) ~ median[F(x[k])].
          The resulting quantile estimates are approximately median-unbiased
          regardless of the distribution of x.
          (**R** type 8)
        - (3/8,3/8): ``p(k) = (k-3/8)/(n+1/4)``: Blom.
          The resulting quantile estimates are approximately unbiased
          if x is normally distributed
          (**R** type 9)
        - (.4,.4)  : approximately quantile unbiased (Cunnane)
        - (.35,.35): APL, used with PWM

    """

    k = np.arange(1, n+1 , 1)
    return 1 - (k - alphap)/(n + 1 - alphap - betap)



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




def distPlot(data, frozenDist=None, ax=None,
             label=None, labelFit=None, marker="+", noData = False,
             order = 1, alpha_ci = None, period=None, 
             alphap = 0.0, betap = 0.0,**kwargs ) :
    """
    Plot parametric distribution together with data
    """

    from matplotlib import pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)
    
    prob = probN(n, alphap = alphap , betap = betap)

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
        ax.legend( loc = 1 )
    ax.grid(True)

    return ax
