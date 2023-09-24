"""
A starting code for a logistic regression model.
"""

from numpy import *
import numpy as np
from random import randrange
class Logistic:
    """
    This class is for the logistic regression model implementation 
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

    def sigmoid(self, z):

        return 1 / (1 + np.exp(-z))

    def predict(self, X):
        y_pred = np.round(self.sigmoid(np.dot(X,self.w)+self.bias)).astype(int)

        return y_pred


    def train_GA(self, X, Y, iteration=1000):
        X = np.array(X)
        self.a, self.b = X.shape
        self.w= np.zeros(self.b)
        #self.w = np.random.uniform(-1/self.b,1/self.b,(self.b,))
        self.bias = 0
        for i in range(iteration):
            pred_y=self.sigmoid(np.dot(X,self.w)+self.bias)
            #print(np.round(self.sigmoid(np.dot(X,self.w)+self.bias)).astype(int))
            self.w = self.w + self.eta*np.dot(X.T, Y-pred_y)/self.a
            self.bias = self.bias+self.eta * np.sum(Y-pred_y)/self.a
        return self.w, self.bias

    def train_SGA(self, X, Y):
        X = np.array(X)
        self.a, self.b = X.shape
        self.w= np.zeros(self.b)
        #self.w = np.random.uniform(-1/self.b,1/self.b,(self.b,))
        self.bias = 0
        for i in range(self.iter):
            index = randrange(self.a)
            pred_y=self.sigmoid(np.dot(X[index],self.w)+self.bias)
            self.w = self.w + self.eta*np.dot(X[index].T, Y[index]-pred_y)
            self.bias = self.bias + self.eta * (Y[index]- pred_y)
        return self.w,self.bias



    def train_reg_SGA(self, X, Y, ):
        X = np.array(X)
        self.a, self.b = X.shape
        self.w = np.zeros(self.b)
        #self.w = np.random.uniform(-1/self.b,1/self.b,(self.b,))
        self.bias = 0
        for i in range(self.iter):
            index = randrange(self.a)
            pred_y=self.sigmoid(np.dot(X[index],self.w)+self.bias)
            self.w = self.w + self.eta * (np.dot(X[index].T, Y[index] - pred_y)-self.lam*self.w)
            self.bias = self.bias + self.eta * (Y[index]- pred_y)
        return self.w, self.bias



                
