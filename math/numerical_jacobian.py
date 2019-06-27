import numpy as np
from scipy.optimize.slsqp import approx_jacobian
from multiprocessing import Pool
from functools import partial

def jac_i( i, func, x0, f0, epsilon, *args):
    dx = np.zeros( x0.shape )
    dx[i] = epsilon
    return (func(*((x0+dx,)+args)) - f0)/epsilon

def approx_jacobian_n(x, func, epsilon, n=1,*args):
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
    f0 = np.atleast_1d(func(*((x0,)+args)))
    jac = np.zeros([n, len(f0)])
    p = Pool(n)
    for i, res in enumerate(p.map( partial( jac_i, func = func, x0=x0, f0=f0, epsilon=epsilon,*args) , list(range(n))  )) :
        jac[i] = res
    return jac.transpose()



def test_fun( x ):
    y = np.empty(x.shape)
    y[:] = x
    y[:-1] = x[:-1]*2
    return y


if __name__ == "__main__" :
    print (approx_jacobian( np.array([1,2,3,4]), test_fun, 0.001 ))
    print (approx_jacobian_n( np.array([1,2,3,4]), test_fun, 0.001,  n = 4 ))

