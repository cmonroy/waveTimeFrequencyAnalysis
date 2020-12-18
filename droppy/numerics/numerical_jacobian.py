import numpy as np
from scipy.optimize.slsqp import approx_jacobian
from multiprocessing import Pool
from functools import partial

def jac_i( i, func, x0, f0, epsilon, *args):
    dx = np.zeros( x0.shape )
    dx[i] = epsilon
    return (func(*((x0+dx,)+args)) - f0)/epsilon

def approx_jacobian_n(x, func, epsilon, fx = None, nproc=1, *args):
    """
    Approximate the Jacobian matrix of a callable function.

    Parameters
    ----------
    x : array_like
        The state vector at which to compute the Jacobian matrix.
    func : callable f(x,*args)
        The vector-valued function.
    epsilon : float
        The perturbation used to determine the partial derivatives.
    n : int
        The number of core to use for the gradient calculation
    args : sequence
        Additional arguments passed to func.

    Returns
    -------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length
    of the outputs of `func`, and ``lenx`` is the number of elements in
    `x`.

    Notes
    -----
    The approximation is done using forward differences.

    """
    n = len(x)
    x0 = np.asfarray(x)

    if fx is None :
        f0 = np.atleast_1d(func(*((x0,)+args)))
    else :
        f0 = np.atleast_1d(fx)

    jac = np.zeros([n, len(f0)])

    if nproc > 1 :
        p = Pool(nproc)
        for i, res in enumerate(p.map( partial( jac_i, func = func, x0=x0, f0=f0, epsilon=epsilon,*args) , list(range(n))  )) :
            jac[i] = res
        p.close()  
        p.join()  
    else :
        dx = np.zeros(len(x0))
        for i in range(len(x0)):
            dx[i] = epsilon
            jac[i] = (func(*((x0+dx,)+args)) - f0)/epsilon
            dx[i] = 0.0



    return jac.transpose()











