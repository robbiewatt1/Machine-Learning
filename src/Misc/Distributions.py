import jax.numpy as np
import math
import random as rand


class Distribution:
    """
    Base class for all distributions
    """

    def density(self, x):
        raise NotImplementedError("Stop using this base class!")

    def sample(self, n):
        """
        Sample n  random numbers from distribution
        :param n: Number of samples
        :return:  Numpy array of samples
        """
        return np.array([x for x in self.generate(n)])

    def generate(self, end):
        raise NotImplementedError("Stop using this base class!")


class Uniform(Distribution):
    """
    Basic uniform distribution class
    """

    def __init__(self, low, high):
        """
        :param low: Lower bound of sample
        :param high: Upper bound of sample
        """
        self.low = low
        self.high = high

    @staticmethod
    def density_static(x, low, high):
        """
        Uniform density function
        :param x: Position
        :param low: Lower bound
        :param high Upper bound
        :return: 1/h if inside 0.5h of mu
        """
        if low < x < high:
            return 1.0 / (low - high)
        else:
            return 0

    def density(self, x):
        """
        Uniform density function
        :param x: position
        :return: 1/h if inside 0.5h of mu
        """
        if self.low < x < self.high:
            return 1.0 / (self.low - self.high)
        else:
            return 0

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


class Binomial(Distribution):
    """
    Basic binomial distribution class
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
        self.cdf[0] = self.density(0)
        for i in range(1, n):
            self.cdf[i] = self.density(i) + self.cdf[i-1]

    @staticmethod
    def density_static(x, n, p):
        """
        :param x: Number of successes
        :param n: Number of attempts
        :param p: Chance of success
        :return: binomial density at x
        """
        k_fac = math.factorial(x)
        n_fac = math.factorial(n)
        nk_fac = math.factorial(n - x)
        return n_fac / (k_fac * nk_fac) * (1. - p) ** (n - x) \
            * p ** x

    def density(self, x):
        """
        :param x: Number of successes
        :return: binomial density at x
        """
        k_fac = math.factorial(x)
        n_fac = math.factorial(self.n)
        nk_fac = math.factorial(self.n - x)
        return n_fac / (k_fac * nk_fac) * (1. - self.p) ** (self.n - x) \
            * self.p ** x

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


class Gaussian1d(Distribution):
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

    @staticmethod
    def density_static(x, mu, sigma):
        """
        Gaussian density function
        :param x: position
        :param mu: mean
        :param sigma: sigma
        :return: gaussian density
        """
        return (2.0 * np.pi * sigma ** 2.0) ** -0.5 \
            * np.exp(-(x - mu) ** 2.0 / (2.0 * sigma ** 2.0))

    def density(self, x):
        """
        Gaussian density function
        :param x: position
        :return: gaussian density
        """
        return (2.0 * np.pi * self.sigma ** 2.0) ** -0.5 \
            * np.exp(-(x - self.mu) ** 2.0 / (2.0 * self.sigma ** 2.0))

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


class GaussianNd(Distribution):
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
        self.covar = covar
        self.size = mu.size
        self.cholesky = np.linalg.cholesky(covar)
        self.gauss_1d = Gaussian1d(0, 1)

        # Calculate once to speed up density
        self.covar_det = np.linalg.det(self.covar)
        self.covar_inv = np.linalg.inv(self.covar)

    @staticmethod
    def density_static(x, mu, covar):
        """
        Gaussian density function
        :param x: position
        :param mu: mean
        :param covar: covariance
        :return: gaussian density
        """
        return np.exp(-0.5 * (x - mu) @ np.linalg.inv(covar) @ (x - mu)) \
            * ((2.0 * np.pi) ** x.size * np.linalg.det(covar)) ** -0.5

    def density(self, x):
        """
        Multivariate Gaussian density function
        :param x: position
        :return: gaussian density
        """
        return np.exp(-0.5 * (x - self.mu) @ self.covar_inv @ (x - self.mu)) \
            * ((2.0 * np.pi) ** self.size * self.covar_det) ** -0.5

    def log_density(self, x):
        """
        Multivariate Gaussian log density function
        :param x: position
        :return: gaussian density
        """
        return -0.5 * (x - self.mu) @ self.covar_inv @ (x - self.mu) \
            + np.log(((2.0 * np.pi) ** self.size * self.covar_det) ** -0.5)

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


def beta():
    pass
