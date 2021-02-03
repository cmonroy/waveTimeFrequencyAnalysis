import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import beta

def probN( n, alphap = 0.0, betap = 0.0):
    """Return exceedance probability of the ranked data

    (inverse of scipy.stats.mstats.mquantiles)


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




def rpPlot(data, duration, frozenDist=None, ax=None,
             label=None, labelFit=None, marker="+", noData = False,
             order = 1, alphap = 0.0, betap = 0.0,**kwargs ) :
    """
    Plot parametric distribution together with data, versus return period

    Parameters
    ----------
    data : np.ndarray
        Data points
    duration : float
        Data duration
    frozenDist : scipy.stats.rv_continouse, optional
        Analytical distribution to be plotted. The default is None.
    ax : matplotlib axes object, optional
        Where to plot. The default is None.
    label : str, optional
        data label. The default is None.
    labelFit : str, optional
        fit label. The default is None.
    marker : str, optional
        Data marker. The default is "+".
    noData : bool, optional
        Do not plot data. The default is False.
    alphap : float, optional
        Value used to calculate the emprical distribution. The default is 0.0.
    betap : float, optional
        Value used to calculate the emprical distribution. The default is 0.0.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : matplotlib axes object,
        The plot.

    """


    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)

    prob = probN(n, alphap = alphap , betap = betap)
    rp = -duration / (n * np.log( 1 - prob ))

    if not noData:
        ax.plot( rp , np.sort(data), marker=marker, label=label, linestyle="", **kwargs)

    if frozenDist is not None:
        ax.plot(rp , frozenDist.isf(prob), "-", label = labelFit, **kwargs)

    ax.set_xscale("log")
    ax.set_xlabel("Return period" )

    if label is not None or labelFit is not None :
        ax.legend( loc = 1 )

    return ax


def distPlot(data, frozenDist=None, ax=None,
             label=None, labelFit=None, marker="+", noData = False,
             order = 1, alpha_ci = None, period=None,
             alphap = 0.0, betap = 0.0,**kwargs ) :
    """
    Plot parametric distribution together with data

    Parameters
    ----------
    data : np.ndarray
        Data points
    frozenDist : scipy.stats.rv_continouse, optional
        Analytical distribution to be plotted. The default is None.
    ax : matplotlib axes object, optional
        Where to plot. The default is None.
    label : str, optional
        data label. The default is None.
    labelFit : str, optional
        fit label. The default is None.
    marker : str, optional
        Data marker. The default is "+".
    noData : bool, optional
        Do not plot data. The default is False.
    order : TYPE, optional
        DESCRIPTION. The default is 1.
    alpha_ci : float, optional
        Confidence interval size. The default is None.
    period : float, optional
        Period to plot data in exceedance rate. The default is None.
    alphap : float, optional
        Value used to calculate the emprical distribution. The default is 0.0.
    betap : float, optional
        Value used to calculate the emprical distribution. The default is 0.0.
    **kwargs : TYPE
        DESCRIPTION.

    Returns
    -------
    ax : matplotlib axes object,
        The plot.

    """
    from matplotlib import pyplot as plt

    if ax is None:
        fig, ax = plt.subplots()
    n = len(data)

    prob = probN(n, alphap = alphap , betap = betap)

    if period is not None:
        conv = 3600. / period  # Convert from probability to events rate
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


def cdf_from_edges_and_count(edges, count) :
    return edges[1:-1], (np.cumsum(count) / np.sum(count))[:-1]


def distPlot_bins(edges, count, frozenDist, label=None, labelFit=None, marker="+", noData = False, ax = None ) :

    if ax is None:
        fig, ax = plt.subplots()

    v_ , cdf_ = cdf_from_edges_and_count(edges, count)
    ax.plot( v_ , 1-cdf_ , "+" )

    prob = np.logspace( -np.log10(np.sum(count))  , np.log10( 1 - 1/np.sum(count)) , 500)

    if frozenDist is not None:
        ax.plot( frozenDist.isf(prob), prob, "-", label = labelFit)

    ax.set_ylabel( "Exceedance probability" )
    ax.set_yscale("log")
    return ax


def distPlot_bins_pdf(edges, count, frozenDist, label=None, labelFit=None, marker="+", noData = False,
                      color_fit = "darkorange", ax = None ) :

    if ax is None:
        fig, ax = plt.subplots()

    ax.bar( edges[:-1], count / (np.sum(count) / (edges[-1]-edges[0]) ),
            width = np.diff(edges), align = "edge", alpha = 0.5, color = "blue")

    if frozenDist is not None:
        c = 0.5*(edges[1:] + edges[:-1])
        ax.plot( c , frozenDist.pdf(c), "-", label = labelFit, color = color_fit, linewidth = 2)

    ax.set_ylabel( "Probability density" )
    ax.set_yscale("log")
    return ax


