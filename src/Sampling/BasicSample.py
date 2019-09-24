import math
import numpy as np
import random as rand
from ..Misc.Distributions import binomial


class Sample:
    """
    Base class for all random sampling methods
    """

    def sample(self, n):
        """
        Sample n  random numbers from distribution
        :param n: Number of samples
        :return:  Numpy array of samples
        """
        return np.array([x for x in self.generate(n)])

    def generate(self, end):
        raise NotImplementedError("Stop using this base class!")


class Uniform(Sample):
    """
    Basic wrapper class around numpy uniform
    """

    def __init__(self, low, high):
        """
        :param low: Lower bound of sample
        :param high: Upper bound of sample
        """
        self.low = low
        self.high = high

    def generate(self, end=1e6):
        """
        Generates a sequence of random uniforms
        :param end: end of generator
        :return generator of random numbers
        """

        i = 0
        while True:
            i += 1
            yield rand.uniform(self.low, self.high)
            if i == end:
                break


class Binomial(Sample):
    """
    Class for sampling from a binomial distribution.
    """

    def __init__(self, n, p):
        """
        :param n: Number of tries
        :param p: Probability of event
        """
        if not isinstance(n, int) or n < 0:
            raise ValueError("n must be positive int!")
        if p < 0. or p > 1.:
            raise ValueError("p must be probability!")
        self.n = n
        self.p = p

        # generate cdf
        self.cdf = np.zeros(n)
        self.cdf[0] = binomial(n, p, 0)
        for i in range(1, n):
            self.cdf[i] = binomial(n, p, i) + self.cdf[i-1]

    def generate(self, end=1e6):
        """
        Generates a sequence of binomial samples
        :param end: End of generator
        :return Generator of binomial numbers
        """

        i = 0
        while True:
            i += 1
            r = rand.uniform(0., 1.)
            for k, bound in enumerate(self.cdf):
                if bound > r:
                    yield k
                    break
            if i == end:
                break


class Gaussian1d(Sample):
    """
    Gaussian random number generator Box–Muller transform
    """

    def __init__(self, mu, sigma):
        """
        :param mu: Mean of gaussian
        :param sigma: standard deviation of gaussian
        """
        self.mu = mu
        self.sigma = sigma

    def generate(self, end=1e6):
        """
        Generates a sequence of random gaussian numbers
        :param end: end of generator
        :return generator of random numbers
        """
        i = 0
        while True:
            r1 = rand.uniform(0., 1.)
            r2 = rand.uniform(0., 1.)
            gauss1 = (-2.0 * math.log(r1))**0.5 * math.cos(2. * math.pi * r2)
            i += 1
            yield (gauss1 - self.mu) / self.sigma
            if i == end:
                break
            gauss2 = (-2.0 * math.log(r1))**0.5 * math.sin(2. * math.pi * r2)
            i += 1
            yield (gauss2 - self.mu) / self.sigma
            if i == end:
                break

    @staticmethod
    def single_sample(mu, sigma):
        r1 = rand.uniform(0., 1.)
        r2 = rand.uniform(0., 1.)
        return ((-2.0 * math.log(r1)) ** 0.5 * math.cos(2. * math.pi * r2) - mu) / sigma


class GaussianNd(Sample):
    """
    Multivariate Gaussian random number might fail semideninite
    matrix.
    """

    def __init__(self, mu, covar):
        """
        :param mu: Numpy array of mean of gaussian
        :param covar: Covariance matrix of gaussian
        """
        # Check inputs are good
        if not isinstance(mu, np.ndarray):
            raise ValueError("mu not numpy array!")
        if not isinstance(covar, np.ndarray):
            raise ValueError("mu not numpy array!")
        if len(mu.shape) != 1 or len(covar.shape) != 2 \
                or mu.shape[0] != covar.shape[0] \
                or mu.shape[0] != covar.shape[1]:
            raise ValueError("Inputs bad shape: {} and {}".format(mu.shape, covar.shape))

        self.mu = mu
        self.sigma = covar
        self.cholesky = np.linalg.cholesky(covar)
        self.gauss_1d = Gaussian1d(0, 1)
        print(self.cholesky)

    def generate(self, end=1e6):
        """
        Generates a sequence of random gaussian numbers
        :param end: End of generator
        :return Generator of random numbers
        """
        i = 0
        while True:
            # Generate standard gaussian samples
            z = self.gauss_1d.sample(self.mu.shape[0])

            i += 1
            yield self.mu + self.cholesky @ z
            if i == end:
                break

    @staticmethod
    def single_sample(mu, covar):
        cholesky = np.linalg.cholesky(covar)
        gauss_1d = Gaussian1d(0, 1)
        z = gauss_1d.sample(mu.shape[0])
        return mu + cholesky @ z
