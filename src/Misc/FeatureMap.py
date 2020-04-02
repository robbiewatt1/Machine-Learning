import numpy as np


class FeatureMap:
    """
    Feature mapping class. Used to map the input space into a higher dimension
    allowing for a more complex model to be fitted. Oly supports a single
    initial feature. By default uses polynomials.
    """
    def __init__(self, function, parameter):
        """
        Constructor sets mapping for class
        :param function: Base function for mapping
        :param paramter: First paramter of mapping function
        """
        self.function = function
        self.parameter = parameter

    def map(self, x):
        """
        Method for mapping to higher dimensional space
        :param x: Input array of length N
        :return: Output array of shape N x D of mapped input
        """

        mapped_array = np.zeros([x.size, len(self.parameter)])
        for i, element in enumerate(x):
            mapped_array[i, :] = [self.function(param, element)
                                  for param in self.parameter]
        return mapped_array

