import numpy as np


class BayesianRegression:

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Constructor for Bayesian Regression. Here we assume a gaussian prior
        on the weights with mean 0 and precision beta. We also assume a gaussian
        likelihood with know precision beta.
        :param alpha: precision of prior
        :param beta:  precision of data
        """
        self.alpha = alpha
        self.beta = beta
        self.input = None
        self.target = None
        self.weights_mean = None
        self.weights_precision = None

    def fit(self, x, y):
        """
        Sets the training data for the Bayesian regression. Also calculate postirior
        for weights, which is also Gaussian.
        :param x: N x M numpy array of input training data, where N is number of
            data points and M is the number of input features
        :param y:  N x 1 numpy array of target training data.
        """

        # Check if first time data has been added
        if self.input is None:
            # Check that training data and target are correct shape.
            if len(x.shape) != 2:
                raise Exception("Input shape error! input x must have shape"
                                "(N x M). Reshape x.reshape(-1, 1) for single"
                                "feature data or x.reshape(1, -1) for one data point")
            if x.shape[0] != y.shape[0]:
                raise Exception("Number of samples in x and y do not match!")
        else:
            # Check training set is same shape is previous data
            if x.shape[1] != self.input.shape[1] - 1:
                raise Exception("Input shape error! Input shape does not match"
                                "previous training data.")
            if x.shape[0] != y.shape[0]:
                raise Exception("Number of samples in x and y do not match!")

        # Add extra dimension to input for the offset
        x = np.hstack((np.full([x.shape[0], 1], 1), x))
        y = y.reshape(-1, 1)

        # Add data to data set
        if self.input is None:
            self.input = x
            self.target = y
        else:
            self.input = np.vstack((self.input, x))
            self.target = np.vstack((self.target, y))

        # Calculate weight mean and precision
        if self.weights_precision is None:
            self.weights_precision = self.alpha * np.identity(x.shape[1])\
                                     + self.beta * x.T @ x
            self.weights_mean = np.linalg.inv(self.weights_precision)\
                                @ (self.beta * x.T @ y)
        else:
            precision_old = np.copy(self.weights_precision)
            self.weights_precision = self.weights_precision\
                                     + self.beta * x.T @ x
            self.weights_mean = np.linalg.inv(self.weights_precision)\
                                @ (precision_old @ self.weights_mean
                                   + self.beta @ x.T @ y)

    def predict(self, x):
        """
        Calculates predicted values from the Bayesian regression.
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
        mean = x @ self.weights_mean

        print(np.linalg.inv(self.weights_precision).shape)
        print(x.T.shape)
        print((np.linalg.inv(self.weights_precision) @ x.T).shape)
        data_noise = np.sum(x @ np.linalg.inv(self.weights_precision) @ x.T, axis=1)
        var = 1.0 / self.beta + np.where(data_noise > 0, data_noise, 0)

        return mean, var
