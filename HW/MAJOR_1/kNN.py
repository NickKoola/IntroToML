from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd

class kNN(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors: int = 3):
        self.n_neighbors = n_neighbors

    def fit(self, X, Y):
        self.X_train = np.copy(X)
        self.Y_train = np.copy(Y)
        return self

    def predict(self, X):
        distances = cdist(X, self.X_train, 'euclidean')

        nearest_indices = np.argpartition(distances, self.n_neighbors, axis=1)[:, :self.n_neighbors]

        nearest_labels = self.Y_train[nearest_indices]

        predictions = np.sign(np.sum(nearest_labels, axis=1))

        return predictions