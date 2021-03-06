import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split

from classification.KNN.algorithm.knn_scratch import KNN
from classification.KNN.algorithm.knn_tuning import KNN_tuning


class DataVisualization:
    def __init__(self, df) -> None:
        self.df = df

    def plot_all(self):
        self.petal_length_vs_petal_width()
        self.sepal_length_vs_sepal_width()
        self.pair_wise_rel()
        self.correlation_matrix()

    def petal_length_vs_petal_width(self):
        sns.scatterplot(x="petal length (cm)", y="petal width (cm)", hue='species', data=self.df)
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.savefig(os.path.join('plots', 'petal_length_vs_petal_width.jpg'))

    def sepal_length_vs_sepal_width(self):
        sns.scatterplot(x="sepal length (cm)", y="sepal width (cm)", hue='species', data=self.df)
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.savefig(os.path.join('plots', 'sepal_length_vs_sepal_width.jpg'))

    @staticmethod
    def pair_wise_rel():
        sns.pairplot(df, hue='species')
        plt.savefig(os.path.join('plots', 'pair_plot.jpg'))

    @staticmethod
    def correlation_matrix():
        corr_df = df.replace({'setosa': 0, 'versicolor': 1, 'virginica': 2})
        plt.figure(dpi=100, frameon=False)
        sns.heatmap(corr_df.corr(), annot=True)
        plt.savefig(os.path.join('plots', 'correlation_matrix.jpg'))


if __name__ == "__main__":
    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target_names[iris.target]

    data_viz = DataVisualization(df=df)
    data_viz.plot_all()

    df.replace({'setosa': 0, 'versicolor': 1, 'virginica': 2}, inplace=True)
    y = df.pop('species')

    X_train, X_test, y_train, y_test = train_test_split(
        df.to_numpy(), y.to_numpy(), test_size=0.2, random_state=312
    )

    K = [3, 5, 7, 10, 30, 100]

    best_k = KNN_tuning(K=K, X_train=X_train, y_train=y_train, distance_metrix="minkowski", p=2)
    knn_clf = KNN(k=best_k)
    knn_clf.fit(X_train=X_train, y_train=y_train)
    predictions = knn_clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print("Accuracy on test data", acc)
