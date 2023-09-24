"""
A starting code for a linear regression model.
"""

from numpy import *
import numpy as np

class Linear:
    """
    This class is for the linear regression model implementation.  
    """

    w = None

    def __init__(self):
        
        self.w = None
        self.eta = 1.0
        self.lam = 0.0
        self.epoch = 1000

    def setEta(self, etaVal):
        self.eta = etaVal

    def setLam(self, lamVal):
        self.lam = lamVal

    def setEpoch(self, nepoch):
        self.epoch = nepoch

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False
    def predict_gd(self, X):
        return np.dot(X,self.w)+self.bias
    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE
        X=np.array(X)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        predict = np.dot(X,self.w)
        return predict

    def train_CFS(self, X, Y):
        """
        Build a vanilla linear regressor by closed-form solution.
        """
        X = np.array(X)
        temp = np.ones((X.shape[0], 1))
        X = np.concatenate((temp, X), axis=1)
        self.w = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))

        return self.w

    def train_ridge_CFS(self, X, Y):
        """
        Build a ridge regressor by closed-form solution.
        """
        X = np.array(X)
        temp = np.ones((X.shape[0], 1))
        X = np.concatenate((temp, X), axis=1)
        I=np.eye(30)
        I[0][0]=0
        self.w = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.lam  * I), np.matmul(X.T, Y))
        return self.w
    def train_ridge_GD(self, X, Y, iteration):
        """
        Build a ridge regressor by gradient descent algorithm.
        """
        X = np.array(X)
        self.a, self.b = X.shape

        self.bias=0
        self.w = np.zeros(self.b)
        for i in range( iteration):
            Y_pred= self.predict_gd(X)



            dw = (- (2 * (X.T).dot(Y - Y_pred)) +
               (2 *self.lam * self.w)) / self.a

            db = - 2 * np.sum(Y - Y_pred) / self.a

            self.w = self.w -self.eta*dw

            self.bias = self.bias -self.eta*db
        return self.w
