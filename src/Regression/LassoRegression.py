import numpy as np
from ..Optimisation.Descent import CoordinateDescent


class LassoRegression:

    def __init__(self):
        """
        Constructor for lasso regression. Here we just define
        the input and target variables
        """
        self.input = None
        self.target = None
        self.weights = None
        self.variance = None

    def fit(self, x, y, alpha):
        """
        Sets the training data for the Lasso regression and also calculates the
        weights matrix and the variance.
        :param x: N x M numpy array of input training data, where N is number of
            data points and M is the number of input features
        :param y:  N x 1 numpy array of target training data.
        :param alpha: coefficient of the lasso term
        """

        # Check that training data and target are correct shape.
        if len(x.shape) != 2:
            raise Exception("Input shape error! input x must have shape"
                            "(N x M). Reshape x.reshape(-1, 1) for single"
                            "feature data or x.reshape(1, -1) for one data point")
        if x.shape[0] != y.shape[0]:
            raise Exception("Number of samples in x and y do not match!")

        # Add extra dimension to input for the offset
        self.input = np.hstack((np.full([x.shape[0], 1], 1.0), x))
        self.target = y.reshape(-1, 1)

        def objective(w):
            w = w.reshape(-1, 1)
            return 1.0 / (2.0 * self.input.shape[0])\
                   * (self.target - self.input @ w).T @ (self.target - self.input @ w)\
                   + 0.5 * alpha * np.sum(np.abs(w))

        weights_init = np.full([self.input.shape[1]], 1.0)
        descent = CoordinateDescent(objective, weights_init)
        self.weights = descent.optimise()

        # calculate variance
        self.variance = (self.target.T @ self.target
                         - self.target.T @ self.input @ self.weights
                         - self.weights.T @ self.input.T @ self.target
                         + self.weights.T @ self.input.T @ self.input
                         @ self.weights) / x.shape[0]
        print(self.weights)

    def predict(self, x):
        """
        Calculates predicted values from the Least squares regression.
        :param x: N x M numpy array of test points.
        :return mean, var: Returns the mean and variance where mean = w^T x
            and variance is always self.var
        """

        # Check that training data and target are correct shape.
        if len(x.shape) != 2:
            raise Exception("Input shape error! input x must have shape"
                            "(N x M). Reshape x.reshape(-1, 1) for single"
                            "feature data or x.reshape(1, -1) for one data point")
        x = np.hstack((np.full([x.shape[0], 1], 1.0), x))
        mean = x @ self.weights
        var = np.full(mean.size, self.variance)

        return mean, var
