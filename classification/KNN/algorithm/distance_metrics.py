import numpy as np


class DistanceMetrics:
    def __init__(self):
        pass

    @staticmethod
    def manhattan_distance_mtx(test_input, train_inputs):
        """
        Manhattan distance measure also known as L1 Norm
        => SUM(|X-Y|)
        :param test_input:
        :param train_inputs:
        :return:
        """
        return np.sum(np.abs(train_inputs - test_input), axis=1)

    @staticmethod
    def euclidean_distance_mtx(test_input, train_inputs):
        """
        Euclidean distance measure also known as L2 Norm
        => SQRT(SUM((X-Y)^2))

        :param test_input:
        :param train_inputs:
        :return: distance from training inputs to test input
        """
        abs_diff = np.abs(train_inputs - test_input)
        return np.sqrt(np.sum(np.power(abs_diff, 2), axis=1))

    @staticmethod
    def minkowski_distance_mtx(test_input, train_inputs, p: int):
        """
        Minkowski distance is genarlization of both L1 & L2.

        => (SUM((X-Y)^P))^(1/P)
        :param test_input:
        :param train_inputs:
        :return:
        """
        abs_diff = np.abs(train_inputs - test_input)
        return np.power(np.sum(np.power(abs_diff, p), axis=1), 1 / p)
