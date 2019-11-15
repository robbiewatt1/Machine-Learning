import numpy as np
from scipy.spatial import KDTree
from scipy.special import gamma
import matplotlib.pyplot as plt
from ..Tools.Decorator import lazy_property
from .Distributions import Gaussian1d
from .Distributions import Uniform


class DensityEstimation:
    """
    Base class for density esimation
    """

    def __init__(self, data):
        """
        :param data: N x D Numpy array of input data.
        """
        self.data = data

        # check data is right shape
        if len(self.data.shape) != 2:
            raise TypeError("Data should be N x D array")

    def density(self, x):
        pass

    def plot_1d(self):
        """
        Method to plot the density over the sample range for 1d data
        """
        try:
            low_lim = min(self.data)
            high_lim = max(self.data)
            x = np.linspace(low_lim, high_lim, 1000)
            dens = self.density(x[:,None])
            fig, ax = plt.subplots()
            ax.plot(x, dens)
            ax.set(xlabel='x', ylabel='Density')
            ax.grid()
        except IndexError:
            print("Error: plot only works for 1D data")


class KernelEstimation(DensityEstimation):
    """
    Class used to preform density estyimation on data sample. Only works for 1D
    at the momentum but I have plans to extend this with multivariate
    distributions
    """

    def __init__(self, data, kernel="gaussian", band_width=1.0):
        """
        :param data: N x D Numpy array of input data.
        :param kernel: Function or string for kernel type
        :param band_width: Band with of kernel
        """
        super(KernelEstimation, self).__init__(data)
        self.kernel = kernel
        self.band_width = band_width

        # If user defined kernel is passed
        if callable(self.kernel):
            self.kernel = kernel
        elif kernel == "gaussian":
            self.kernel = Gaussian1d.density_static
        elif kernel == "uniform":
            self.kernel = Uniform.density_static
        else:
            raise TypeError("Unsupported kernel {}".format(kernel))

    def density(self, x):
        """
        Method to return the estimated PDF which generated the data
        :param x: Region of interest
        :return: Density qwith same shape as x
        """
        # sum all kernels located at data point
        dens = np.zeros_like(x)
        for element in self.data:
            dens += self.kernel(x, element, self.band_width)
        return dens / self.data.shape[0]


class KnnEstimation(DensityEstimation):
    """
    Class to preform knn density esimator. Uses a scipy kd tree to find nn
    faster.
    """
    def __init__(self, data, k=1):
        """
        :param data: N x K numpy array of data
        :param k: Order of nearest neighbor
        """
        super(KnnEstimation, self).__init__(data)
        self.k = k

    @lazy_property
    def kd_tree(self):
        return KDTree(self.data)

    def density(self, x):
        """
        Method to return the estimated PDF which generated the data
        :param x: N x D numpy array of query points
        :return: Density with same shape as x
        """
        dimensions = x.shape[1]
        unit_volume = gamma(dimensions / 2.0 + 1) * np.pi ** (dimensions / 2.0)
        closest_points, _ = self.kd_tree.query(x, self.k)
        return self.k / (self.data.shape[0] * unit_volume * closest_points[:, self.k-1])

