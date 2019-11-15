import random as rand
import numpy as np
import jax.numpy as jnp
from jax import grad, jit
from ..Misc import Distributions as Dist
import matplotlib.pyplot as plt
from matplotlib import cm


from autograd import make_vjp

class MCMC:
    """
    Base MCMC class, contains some methods for determining and visualising
    the convergence of the Markov chain.
    """

    def sample(self, n):
        """
        Sample random numbers from distribution
        :param n: Number of samples
        :return:  Numpy array of samples
        """
        return np.array([x for x in self.generate(n)])

    def generate(self, end=1e6):
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
        samples = np.array(list(self.generate(end=sample_length)))
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
            plt.subplot((1+dims/2), 2, i+1)
            plt.plot(iter_num, samples[:, i])
        plt.figure(2)
        plt.plot(iter_num, auto_corr)

    def sample_test(self, time=10):
        """
        Function to test if sampling is doing what you think. Will plot contour
        of function and samples being drawn. Currently only works for 2 dimensional
        functions.
        """

        # Get the random number generator
        rand_gen = self.sample(100)

        # plot the function surface
        xy = np.arange(-5, 5, 0.1)
        X, Y = np.meshgrid(xy, xy)
        coords = np.stack((X, Y), axis=2)
        Z = np.zeros_like(X)
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                Z[i, j] = self.func(coords[i, j, :])

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        ax.contour(X, Y, Z, cmap=cm.plasma)
        ax.scatter(rand_gen[:, 0], rand_gen[:, 1])
        print(rand_gen)



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
               gaussian with diagonal covariance. Should take one parameters x
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
                return Dist.GaussianNd(x, covar).sample(1)[0]

            def proposal_function(x, y):
                return Dist.GaussianNd(y, covar).density(x)

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

    def generate(self, end=1e6):
        """
        :param end: End of generator
        :return: Numpy array of samples
        """

        x = self.x_init
        for _ in range(self.burn_off):
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
            hast_ratio = self.func(y) * self.prop_func(x, y) / \
                (self.func(x) * self.prop_func(x, y))
            if min(1.0, hast_ratio) > rand.uniform(0., 1.):
                x = np.copy(y)
            yield x

            if i == end:
                break

class Gibbs:
    pass


class Hamiltonian(MCMC):
    """
    MCMC sampler using the Hamiltonian algorithm. We have set the kenetic term
    to be gaussian. Inherits from Sample so has both a generator and a sampler method.
    """
    def __init__(self, func, q_init, step_size=0.5, path_length=1.0, mass=1.0, gradient=None):
        """
        :param func: Function to be sampled from. Must take a numpy array as input
        :param q_init: Chain starting point
        :param step_size: Length of each integration step
        :param path_length: Total length of integration
        """
        self.func = func
        self.q_init = q_init
        self.step_size = step_size
        self.path_length = path_length

        self.mass = np.identity(self.q_init.size) * mass
        self.inv_mass = np.identity(self.q_init.size) / mass

        self.potential = lambda x: - jnp.log(self.func(x))
        if gradient is None:
            self.grad_func = jit(grad(self.potential))
        else:
            self.grad_func = gradient

        # Preform some input checks
        try:
            self.func(self.q_init)
            self.grad_func(self.q_init)
        except Exception as e:
            raise ValueError("Class input parameters bad. Error is: {}". format(str(e)))

    @staticmethod
    def _leap_frog(func, q_init, p_init, step_size, path_length):
        """
        Leapfrog integration method to find hamiltonian paths
        :param func:        dv/dt function being integrated
        :param p_init:      p starting point
        :param q_init:      q starting point
        :param step_size:   delta step
        :param path_length: total length of path
        :return: (x_final, v_final)
        """
        n_steps = int(path_length / step_size)
        for i in range(n_steps):
            q_new = q_init + p_init * step_size + 0.5 * func(q_init) \
                * step_size * step_size
            p_new = p_init + 0.5 * (func(q_init) + func(q_new)) * step_size
            q_init = np.copy(q_new)
            p_init = np.copy(p_new)
        return q_init, p_init

    def generate(self, end=1e6):
        # Momentum sampler and density
        gauss_dist = Dist.GaussianNd(np.zeros_like(self.q_init), self.mass)
        p_sampler = gauss_dist.generate(end)
        p_log_density = jit(gauss_dist.log_density)

        q_current = np.copy(self.q_init)
        for _ in range(int(end)):
            # Sample momentum and solve dynamics
            p_current = next(p_sampler)
            q_final, p_final = self._leap_frog(self.grad_func, q_current, p_current,
                                               self.step_size, self.path_length)

            # Check Metropolis acceptance criterion
            init_log_p = self.potential(q_current) - p_log_density(p_current)
            final_log_p = self.potential(q_final) - p_log_density(p_final)
            if np.log(rand.uniform(0., 1.)) < init_log_p - final_log_p:
                yield q_final
                q_current = np.copy(q_final)
            else:
                yield q_current

