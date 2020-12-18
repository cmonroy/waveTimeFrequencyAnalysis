from numpy import zeros, vdot, sqrt, abs, sign
from numpy.linalg import norm
import matplotlib.pyplot as plt

def HLCL(x0, func, beta_tol, func_tol, nsearch , debug = False , dx = 1e-3 , relax = 0.5):
    """
    x1, G1 = form(x0, G, beta_tol, gtol)
    solve G(x) = 0 using extension of Hasofer-Lind method with circle and line search
    x0: starting point - should be close enough to the solution...
    G: objective function
    beta_tol: absolute error for beta = norm(x)
    gtol: absolute error for the value of G
    """

    nitermax = 200

    dzeta = 0.2

    n = len(x0)
    dG = zeros(n)
    xk = zeros(n)
    ak = zeros(n)
    y0 = zeros(n)
    y = zeros(n)
    test = zeros(nsearch)


    G1 = func(x0)


    xk = x0

    beta0 = 0.0
    for iiter in range(nitermax):

        beta = norm(xk)
        if iiter>0 and abs(beta-beta0)<beta_tol:
            return xk, G1
        else:
            beta0 = beta

        # gradient
        for i in range(n):
            xk[i] = xk[i] + dx
            dG[i] = (func(xk)-G1) / dx
            xk[i] = xk[i] - dx
        dGnorm = vdot(dG, dG)
        dGx = vdot(dG, xk)

        # circle search
        ak = dG*(dGx-G1)/dGnorm
        anorm = norm(ak)
        y0 = ak
        G0 = G1
        for m in range(nsearch):
            # new point on the circle
            zeta = 0.1 + m*dzeta
            y = zeta*ak + (1.0-zeta)*xk
            y = y*anorm/norm(y)
            # evaluate G
            G1 = func(y)
            test[m] = G1
            # check if G=0 or if minimum is reached
            if m>0:
                if (G0>0.0 and G1>G0) or (G0<0.0 and G1<G0) or (G1*G0<=0.0):
                    break
            G0 = G1
            y0 = y

        if m==nsearch-1 :
            plt.figure()
            plt.grid(True)
            plt.plot(test, label='problem with circle search!')
            plt.legend()
            plt.show()
            exit()


        # line search
        xk = y0
        step0 = 1.0
        step = 1.0 + 0.1*sign(G0)
        for m in range(nsearch):
            # new x value along the line
            y = xk * step
            # evaluate G
            G1 = func(y)
            test[m] = G1
            if abs(G1)<gtol:
                xk = y
                break
            else:
                a = (G1-G0)/(step-step0)
                b = G1 - a*step
                step0 = step
                G0 = G1
                step = -b/a
                # relaxation
                if m>3:
                    step = relax*step + (1.-relax)*step0

        if m==nsearch-1 and debug:
            plt.figure()
            plt.grid(True)
            plt.plot(test, label='problem with line search!')
            plt.legend()
            plt.show()
            exit()

    return xk , G1