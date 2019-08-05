import sys
import random as rand
from .BasicSample import *


class MetroHast(Sample):
    """
    MCMC sampler using the Metropolis-Hastings algorithm. Inherits from Sample
    so has both a generator and a sampler method. Currently assumes symmetric
    proposal.
    """
    def __init__(self, func, x_init, prop_dist=None, burn_off=1000):
        """
        :param func: Function to be sampled from. Must take a numpy array as input
        :param x_init: Numpy array vector of initial point
        :param prop_dist: Proposal distribution, if none then set to gaussian. Should
        be a function of a single parameter x
        :param burn_off: Number of initial samples before accepting
        """

        self.func = func
        self.x_init = x_init
        self.burn_off = burn_off

        if prop_dist is None:
            # We set it to gaussian with identity covar
            covar = np.identity(x_init.size)

            def proposal(x):
                return GaussianNd.single_sample(x, covar)

            self.prop_dist = proposal

        # Preform some input checks
        try:
            self.func(self.x_init)
        except Exception as e:
            print("func cannot accept x_init as an input. Error is: {}". format(str(e)))
            sys.exit(1)

    def generate(self, end=1e6):
        """
        :param end: End of generator
        :return: Numpy array of samples
        """

        # x and y are current and future steps
        # Initial burn off period
        x = self.x_init
        for _ in range(self.burn_off):
            # Draw from proposal distribution
            y = self.prop_dist(x)
            hast_ratio = self.func(y) / self.func(x)
            if min(1.0, hast_ratio) > rand.uniform(0., 1.):
                x = np.copy(y)

        # Now we can yield new samples
        i = 0
        while True:
            i += 1
            # Draw from proposal distribution
            y = self.prop_dist(x)
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
