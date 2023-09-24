"""
A starting code for a logistic regression model.
"""

from numpy import *
import numpy as np
from binary import *
import util
from regression import *

class Logistic(BinaryClassifier):
    """
    This class is for the logistic regression model implementation 
    for multi-class classification problem.
    """

    def __init__(self, lr=0.01, n_iter=10000):
        self.lr = lr
        self.n_iter = n_iter

    def setLambda(self, _lambdaVal):
        self._lambda = _lambdaVal

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def loss(self, h, y):
        return (-y * np.log(h + np.finfo(float).eps) - (1 - y) * np.log(1 - h + np.finfo(float).eps)).mean()

    def probability(self, X):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        return self.sigmoid(X @ self.w)

    def predict(self, X, threshold):
        return self.probability(X) >= threshold

    def train(self, X, Y, lr=0.01):
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        #self.w = np.random.normal(0, 1, size=(X.shape[1]))
        self.w = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            h = self.sigmoid(X @ self.w)
            grad = X.T @ (h - Y) / Y.size
            self.w -= self.lr * grad
            if (i % 1000 == 0):
                print(self.loss(h, Y))
