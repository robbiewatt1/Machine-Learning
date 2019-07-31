import math
import numpy as np
import random as rand


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
        yield None


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
            gauss2 = (-2.0 * math.log(r2))**0.5 * math.sin(2. * math.pi * r1)
            i += 1
            yield (gauss2 - self.mu) / self.sigma
            if i == end:
                break
