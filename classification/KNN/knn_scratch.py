from collections import Counter
import numpy as np


def euclidean_distance(test_input, train_inputs):
    abs_diff = np.abs(train_inputs - test_input)
    return np.sqrt(np.sum(np.power(abs_diff, 2), axis=1))


class KNN:
    def __init__(self, k: int = 3, distance_measurement: str = "euclidean"):
        self.k = k
        self.distance_measurement = distance_measurement

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distance(self, test_input):
        distances = []
        if self.distance_measurement == "euclidean":
            distances = euclidean_distance(
                train_inputs=self.X_train, test_input=test_input
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