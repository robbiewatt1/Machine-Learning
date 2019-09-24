import sys
import random as rand
import numpy as np
import matplotlib.pyplot as plt
from . import BasicSample as BS
from ..Misc import Distributions as Dist
from ..Misc.Misc import vector_autocorrelate


class MCMC(BS.Sample):
    """
    Base MCMC class, contains some methods for determining and visualising
    the convergence of the Markov chain.
    """

    def generate(self, end=1e6, burn=None):
        """
        Overrides the base class generator in sample with extra burn
        parameter for discarding start of chain
        """
        raise NotImplementedError("Stop using this base class!")

    def diagnose(self, sample_length):
        """
        Plots the autocorrolation and trace of the samples from the MCMC
        :param sample_length: Number of samples taken
        """

        # Generate the samples
        samples = np.array(list(self.generate(end=sample_length, burn=1)))
        dims = samples.shape[1]

        # Generate the auto correlation
        n_vectors = len(samples)
        # correlate each component independently
        auto_corr = np.array([np.correlate(samples[:, i], samples[:, i], 'full')
                              for i in range(dims)])[:, n_vectors - 1:]
        # sum the correlations for each component
        auto_corr = np.sum(auto_corr, axis=0)
        # divide by the number of values actually measured and return
        auto_corr /= (n_vectors - np.arange(n_vectors))

        iter_num = np.array(list(range(sample_length)))

        plt.figure(1)
        for i in range(dims):
            print("here")
            plt.subplot((1+dims/2), 2, i+1)
            print(samples[:, i])
            plt.plot(iter_num, samples[:, i])
        plt.figure(2)
        plt.plot(iter_num, auto_corr)


class MetroHast(MCMC):
    """
    MCMC sampler using the Metropolis-Hastings algorithm. Inherits from Sample
    so has both a generator and a sampler method. Currently assumes symmetric
    proposal.
    """
    def __init__(self, func, x_init, prop_samp=None, prop_func=None,
                 burn_off=1000, thinning=1):
        """
        :param func: Function to be sampled from. Must take a numpy array as input
        :param x_init: Numpy vector of initial point
        :param prop_samp: Proposal distribution sampler, if none then set to
        gaussian with diagonal covariance. Should take a single parameter x
        :param prop_func: Proposal function, should take two parameters and
        return a density
        :param burn_off: Number of initial samples before accepting
        :param thinning: Number of jumps in acceptance
        """

        self.func = func
        self.x_init = x_init
        self.burn_off = burn_off
        self.thinning = thinning

        if prop_samp is None and prop_func is None:
            # We set it to gaussian with identity covar
            covar = np.identity(x_init.size)

            def proposal_sample(x):
                return BS.GaussianNd.single_sample(x, covar)

            def proposal_function(x, y):
                return Dist.gaussian_nd(x, y, covar)

            self.prop_samp = proposal_sample
            self.prop_func = proposal_function

        # Check that both prop_samp and prop_func have been passed
        if prop_samp is None and prop_func is not None \
                or prop_samp is not None and prop_func is None:
            raise ValueError("You must supply both a sampler function and a "
                             "density function")

        # Preform some final input checks
        try:
            self.func(self.x_init)
            self.prop_samp(self.x_init)
            self.prop_func(self.x_init, self.x_init)
        except Exception as e:
            raise ValueError("Class input parameters bad. Error is: {}". format(str(e)))

    def generate(self, end=1e6, burn=None):
        """
        :param end: End of generator
        :param burn: Define burn off at generator creation
        :return: Numpy array of samples
        """

        # Check for changed burn parameter
        if burn is None:
            burn = self.burn_off
        # Initial burn off period
        x = self.x_init
        for _ in range(burn):
            # Draw from proposal distribution
            y = self.prop_samp(x)
            hast_ratio = self.func(y) * self.prop_func(x, y) / \
                (self.func(x) * self.prop_func(x, y))
            if min(1.0, hast_ratio) > rand.uniform(0., 1.):
                x = np.copy(y)

        # Now we can yield new samples
        i = 0
        while True:
            i += 1
            # Draw from proposal distribution
            y = self.prop_samp(x)
            hast_ratio = self.func(y) / self.func(x)
            if min(1.0, hast_ratio) > rand.uniform(0., 1.):
                x = np.copy(y)
            yield x

            if i == end:
                break

class Gibbs:
    pass


class Hamiltonian:
    pass
