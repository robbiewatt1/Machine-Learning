import numpy as np
import math


def gaussian1d(x, mu, sig):
    """
    Gaussian density function
    :param x: position
    :param mu: mean
    :param sig: standard deviation
    :return: gaussian density
    """
    return (2.0 * np.pi * sig**2.0)**-0.5 * np.exp(-(x - mu)**2.0 / (2.0 * sig**2.0))


def gaussian_nd(x, mu, covar):
    """
    Multivariate Gaussian density function
    :param x: position
    :param mu: mean
    :param covar: covariance
    :return: gaussian density
    """
    return np.exp(-0.5 * (x - mu) @ np.linalg.inv(covar) @ (x - mu))\
        * ((2.0 * np.pi) ** x.size * np.linalg.det(covar)) ** -0.5


def uniform(x, mu, h):
    """
    Uniform density function
    :param x: position
    :param mu: distribution centre
    :param h: distribution width
    :return: 1/h if inside 0.5h of mu
    """
    return 1.0 / h if np.abs(x - mu) < 0.5 * h else 0


def binomial(n, p, k):
    """
    :param n: Number of events
    :param p: Probability of event
    :param k: Number of successes
    :return: binomial density at k
    """
    k_fac = math.factorial(k)
    n_fac = math.factorial(n)
    nk_fac = math.factorial(n - k)
    return n_fac / (k_fac * nk_fac) * (1. - p)**(n - k) * p**k


def beta():
    pass
