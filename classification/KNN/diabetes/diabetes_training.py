from typing import List
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from classification.KNN.algorithm.knn_scratch import KNN
from classification.KNN.algorithm.knn_tuning import KNN_tuning

data = pd.read_csv('D:\Projects\Machinelearning\learning-ml\data\diabetes\diabetes.csv')
y = data.pop('Outcome')

X_train, X_test, y_train, y_test = train_test_split(
    data.to_numpy(), y.to_numpy(), test_size=0.2, random_state=3
)

K = [1, 3, 5, 7, 10, 100]
best_k = KNN_tuning(K=K, X_train=X_train, y_train=y_train, distance_metrix="minkowski", p=2)
knn_clf = KNN(k=best_k)
knn_clf.fit(X_train=X_train, y_train=y_train)
predictions = knn_clf.predict(X_test)

acc = np.sum(predictions == y_test) / len(y_test)
print("Accuracy on test data", acc)