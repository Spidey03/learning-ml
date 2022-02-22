import numpy as np


def euclidean_distance(test_inputs, train_inputs):
    return np.sqrt(np.sum((train_inputs - test_inputs) ** 2))


class KNN:
    def __init__(self, k: int = 3, distance_measurement: str = "euclidean"):
        self.k = k
        self.distance_measurement = distance_measurement

    def fit(self, X_train, y_train):
        pass

    def calculate_distance(self, test_X):
        distances = []
        if self.distance_measurement == "euclidean":
            distances = euclidean_distance(
                train_inputs=self.X_train, test_inputs=test_X
            )
        return distances

    def predict(self, test_X):
        distances = self.calculate_distance(test_X=test_X)
        indices = np.argsort(distances)

        # TODO: handle distance tie
        k_indices = indices[: self.k]
        k_nearest_neighbors = self.y_train[k_indices]

        # TODO: handle voting tie
        predicted_outputs = self.y_train[k_indices]
        return predicted_outputs
