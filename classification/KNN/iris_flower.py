import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from knn_tuning import KNN_tuning
from knn_scratch import KNN

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=312
)
K = [1, 2, 3, 5, 7, 8, 9, 10, 100]

best_k = KNN_tuning(K=K, X_train=X_train, y_train=y_train)
knn_clf = KNN(k=best_k)
knn_clf.fit(X_train=X_train, y_train=y_train)
predictions = knn_clf.predict(X_test)

acc = np.sum(predictions == y_test)/len(y_test)
print(acc)