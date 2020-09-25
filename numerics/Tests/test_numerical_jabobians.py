
import numpy as np
from droppy.numerics import approx_jacobian_n
from scipy.optimize.slsqp import approx_jacobian


def fun1( x ):
    y = np.empty(x.shape)
    y[:] = x
    y[:-1] = x[:-1]*2
    return y.sum()

def fun2( x ):
    y = np.empty(x.shape)
    y[:] = x
    y[:-1] = x[:-1]*2
    return y


def test_num_jac() :
    x0 = np.array(np.linspace(1,12,4))
    for test_fun in [fun1 , fun2] :
        #print (test_fun)
        ref = approx_jacobian( x0, test_fun, 0.001 )
        app1 = approx_jacobian_n( x0, test_fun, 0.001,  nproc = 1 )
        app2 = approx_jacobian_n( x0, test_fun, 0.001,  nproc = 4 )
        app3 = approx_jacobian_n( x0, test_fun, 0.001, test_fun(x0), nproc = 4 )

        assert( np.isclose(ref , app1).all() )
        assert( np.isclose(ref , app2).all() )
        assert( np.isclose(ref , app3).all() )
    print (ref)
    print (app1)
    print ("Ok")

if __name__ == "__main__" :
    test_num_jac()