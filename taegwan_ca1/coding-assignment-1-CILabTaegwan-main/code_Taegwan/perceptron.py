"""
A starting code for a perceptron.
"""

from numpy import *
import numpy as np

class Perceptron:
    """
    This class is for the perceptron implementation 
    for binary classification problem.
    """

    def __init__(self):
        """
        Initialize our internal state.
        """
        self.w = None
        self.eta = 1.0
        self.lam = 0.0
        self.iter = 1000
        self.thresh = 0.001
        self.bias=None
    def setEta(self, etaVal):
        self.eta = etaVal

    def setLam(self, lamVal):
        self.lam = lamVal
        
    def setMaxiter(self, niter):
        self.iter = niter
        
    def setThreshold(self, threshVal):
        self.thresh = threshVal

    def model(self, x):
        return 1 if (np.dot(self.w, x) +self.bias >= self.thresh) else 0
    def predict(self, X):

        y_pred  = list()
        for x in X:
            sig = self.model(x)
            y_pred.append(sig)
        return np.array(y_pred)

    def train(self, X, Y):
        X = np.array(X)
        self.w = np.ones(X.shape[1])
        self.bias = 0
        for i in range(10):
            for x, y in zip(X, Y):
                y_pred = self.model(x)
                if y == 1 and y_pred == 0:
                    self.w = self.w + self.eta * x
                    self.bias = self.bias + self.eta * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - self.eta * x
                    self.bias = self.bias - self.eta * 1
        return self.w,self.bias

                
