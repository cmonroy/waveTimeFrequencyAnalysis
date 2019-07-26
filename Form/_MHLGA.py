# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 15:18:26 2019

@author: couledhousseine
"""

import time
import numpy as np
import numpy.linalg as LA
import scipy
from droppy.numerics.numerical_jacobian import approx_jacobian_n

# -----------------------------------------------------------------------------

# function to minimize (for scipy tests) --------------------------------------
# function
def f(x):
    return LA.norm(x)**2
# gradient
def df(x):
    return 2*x
# -----------------------------------------------------------------------------

def MHLGA(x0, func, beta_tol, func_tol, itmax, dx = 1e-3, nproc = 1):

    # MHLGA alogorithm for FORM

    # input -------------------------------------------------------------------
    # x0        : starting solution
    # beta_tol  : beta tolerance (absolute error)
    # func_tol  : limit state function tolerance (absolute error)
    # itmax     : maximum number of iterations
    #--------------------------------------------------------------------------

    # parameters
    A     = 10.
    B     = 100.
    alpha = 1.
    m1    = 0.1
    m2    = 0.9
    r1    = 0.5
    r2    = 1.5

    start = time.time()

    xk    = x0
    xk1   = x0
    Gk    = func(x0)

    beta0 = 0.

    for k in range(itmax):

        beta = LA.norm(xk)

        if(abs(beta-beta0) < beta_tol):
            break
        else:
            beta0 = beta

        Gk  = func(xk)

        # gradient
        dGk = approx_jacobian_n(func=func,x=xk,fx=Gk,epsilon=dx,nproc=nproc)[0]

        # direction
        Nk  = LA.norm(dGk)**2
        ak  = np.dot(xk,dGk)/Nk
        dk  = (ak-Gk/Nk)*dGk-xk

        #if(LA.norm(dk) < beta_tol):
        #    break

        if(abs(Gk) < func_tol):
            ck = B
        else:
            ck = A * abs(ak / Gk)

        # line search to minimize merit function
        alphak = alpha
        mk     = 0.5*(LA.norm(xk)**2+ck*Gk**2)
        dmk    = xk + ck*Gk*dGk
        pk     = np.dot(dmk,dk)

        for i in range(itmax):

            xk1  = xk+alphak*dk
            Gk1  = func(xk1)
            mk1  = 0.5*(LA.norm(xk1)**2+ck*Gk1**2)

            if (mk1 - mk > alphak*m1*pk):
                alphak = r1 * alphak
            elif (mk1 - mk < alphak*m2*pk):
                alphak = r2 * alphak
            else:
                break

        xk  = xk1
        Gk  = Gk1

    cpu     = time.time() - start

    print(' ')
    print('-------------------------------------------')
    print('********* FORM - MHLGA algorithm **********')
    print('-------------------------------------------')
    print('Inputs')
    print('    > beta tolerance (absolute error): ',beta_tol)
    print('    > Limit state function tolerance (absolute error): ',func_tol)
    print('Outputs')
    print('    > Nbr of iterations exceeded: ', k == itmax - 1)
    print('    > Iterations: ',k)
    print('    > beta: ',beta)
    print('    > Limit state function: ',Gk)
    print('    > cpu time: ', cpu)
    print(' ')

    return scipy.optimize.OptimizeResult( x = xk, success=True, fun=beta, nit=k , cpu =cpu, constraint = Gk)
