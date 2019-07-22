import numpy as np


def gaussian(x, mu, sig):
    """
    Gaussian density function
    :param x: position
    :param mu: mean
    :param sig: standard deviation
    :return: gaussian density
    """
    return (2.0 * np.pi * sig**2.0)**-0.5 * np.exp(-(x - mu)**2.0 / (2.0 * sig**2.0))


def uniform(x, mu, h):
    """
    Unifrom density function
    :param x: position
    :param mu: distribution centre
    :param h: distribution width
    :return: 1/h if inside 0.5h of mu
    """
    return 1.0 / h if np.abs(x - mu) < 0.5 * h else 0

def binomial():
    pass


def beta():
    pass
