from typing import List

import numpy as np
from sklearn.model_selection import train_test_split

from classification.KNN.knn_scratch import KNN


def KNN_tuning(K: List[int], X_train, y_train):
    accuracy = []
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=0.2, random_state=312
    )
    for k in K:
        knn_clf = KNN(k=k)
        knn_clf.fit(X_train=X_train, y_train=y_train)
        predictions = knn_clf.predict(X_valid)
        acc = np.sum(predictions == y_valid) / len(y_valid)
        accuracy.append(acc)

    best_accuracy_idx = accuracy.index(max(accuracy))
    print(f"Best Accuracy: {accuracy[best_accuracy_idx]} Best K: {K[best_accuracy_idx]}")
    return K[best_accuracy_idx]
