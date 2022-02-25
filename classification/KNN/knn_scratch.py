from collections import Counter

import numpy as np

from classification.KNN.distance_metrics import DistanceMetrics


class KNN:
    def __init__(self, k: int = 3, distance_measurement: str = "euclidean", **kwargs):
        self.k = k
        self.distance_measurement = distance_measurement
        self.p = kwargs.get('p', 1)

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distance(self, test_input):
        distances = []
        distance_mtx = DistanceMetrics()
        if self.distance_measurement == "manhattan":
            distances = distance_mtx.manhattan_distance_mtx(
                train_inputs=self.X_train, test_input=test_input
            )
        elif self.distance_measurement == "euclidean":
            distances = distance_mtx.euclidean_distance_mtx(
                train_inputs=self.X_train, test_input=test_input
            )
        else:
            distances = distance_mtx.minkowski_distance_mtx(
                train_inputs=self.X_train, test_input=test_input, p=self.p
            )
        return distances

    def predict(self, test_X):
        y_pred = [self.predict_for_x(x) for x in test_X]
        return y_pred

    def predict_for_x(self, test_input):
        distances = self.calculate_distance(test_input=test_input)
        indices = np.argsort(distances)

        # TODO: handle distance tie
        k_indices = indices[: self.k]
        k_nearest_neighbors = self.y_train[k_indices]

        # TODO: handle voting tie
        predicted_output = Counter(k_nearest_neighbors).most_common(1)
        return predicted_output[0][0]
