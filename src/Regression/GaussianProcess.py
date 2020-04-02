import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt


class GaussianProcess:

    def __init__(self, kerenl):
        """
        Constructor for Gaussian process class
        :param kerenl: Kernel function of Gaussian process prior. must derive
            from kernel base class
        """
        self.kernel = kerenl
        # Define member data set later
        self.input = None
        self.target = None
        self.chol_gram = None

    def fit(self, x, y):
        """
        Set the training data set for the Gaussian process and calculate the
        gram matrix.
        :param x: N x M numpy array of input training data, where N is number of
            data points and M is the number of input features
        :param y: N x 1 numpy array of target training data.
        """
        # Check that training data and target are correct shape.
        if len(x.shape) != 2:
            raise Exception("Input shape error! input x must have shape"
                            "(N x M). Reshape x.reshape(-1, 1) for single"
                            "feature data or x.reshape(1, -1) for one data point")
        if x.shape[0] != y.shape[0]:
            raise Exception("Number of samples in x and y do not match!")
        self.input = x
        self.target = y.reshape(-1, 1)
        self.chol_gram = np.linalg.cholesky(self.kernel(self.input,
                                                        self.input))

    def predict(self, x):
        """
        Calculates predicted values from the Gaussian process.
        :param x: N x M numpy array of test points.
        :return mean, var: Returns the mean and variance of the Gaussian
            process evaluated at the test points x.
        """
        # Check that training data and target are correct shape.
        if len(x.shape) != 2:
            raise Exception("Input shape error! input x must have shape"
                            "(N x M). Reshape x.reshape(-1, 1) for single"
                            "feature data or x.reshape(1, -1) for one data point")

        k_s = self.kernel(x, self.input)
        k_ss = self.kernel(x, x)
        alpha = solve(self.chol_gram.T, solve(self.chol_gram, self.target))
        beta = solve(self.chol_gram, k_s.T)
        mean = k_s @ alpha
        var = np.diagonal(k_ss - beta.T @ beta).reshape(-1, 1)

        return mean, var

    def


class Kernel:

    def __init__(self, norm, length):
        self.length = length
        self.norm = norm

    def __call__(self, x1, x2):
        kern = np.zeros([x1.shape[0], x2.shape[0]])
        for i, xi in enumerate(x1):
            for j, xj in enumerate(x2):
                kern[i, j] = self.function(xi, xj)
        return kern

    def function(self, x1, x2):
        """
        Kernel function. Takes a single data point at a time
        :param x1: First data point
        :param x2: Second data point
        :return: covariance function
        """
        return self.norm * np.exp(- 0.5 * (x1 - x2).T @ (x1 - x2) / self.length ** 2.0)

if __name__ == "__main__":
    kern = Kernel(2, 1.5)
    gp = GaussianProcess(kern)

    in_data = np.linspace(0, 1, 5)
    out_data = 2.0 * np.sin(2.0 * np.pi * in_data) + (2.0 * np.pi * in_data)**2.0 - 0.2 * (2.0 * np.pi * in_data)**3.0

    gp.fit(in_data.reshape(-1, 1), out_data)

    test = np.linspace(0, 1, 100)

    mean, var = gp.predict(test.reshape(-1, 1))


    in_data2 = np.linspace(0, 1, 100)
    out_data2 = 2.0 * np.sin(2.0 * np.pi * in_data2) + (2.0 * np.pi * in_data2)**2.0 - 0.2 * (2.0 * np.pi * in_data2)**3.0

    print(var.shape)
    print(mean.shape)
    fig, ax = plt.subplots(1,2)
    ax[0].plot(in_data2, out_data2)
    ax[0].plot(test, mean)
    ax[0].scatter(in_data, out_data)
    ax[0].plot(test, mean + var**0.5)
    ax[0].plot(test, mean - var**0.5)
    ax[1].scatter(test, var**0.5)
    plt.show()




