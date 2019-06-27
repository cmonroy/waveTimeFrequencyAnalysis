from scipy.stats import norm

from droppy.pyplotTools import qqplot, qqplot2


if __name__ == "__main__" :
    a = norm().rvs(1000)
    b = norm().rvs(1000)

    qqplot(a, norm())
    qqplot2(a,b, "a", "b")
