"""
A starting code for a ridge regression model.
"""

from numpy import *
import numpy as np
from binary import *
import util
from regression import *

class Ridge(Regression):
    """
    This class is for the ridge regressor implementation.
    """

    w = None
    lambda_val = 0.0

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

    def setLambda(self, lambdaVal):
        self.lambda_val = lambdaVal
        
        return self.lambda_val

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the model
        """
        return self.w

    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE
        X = X.to_numpy()
        predict = np.matmul(X, self.w)
        #util.raiseNotDefined()
        
        return predict

    def train(self, X, Y):
        """
        Build a ridge regressor.
        """
        ### TODO: YOUR CODE HERE
        
        X = X.to_numpy()
        Y = Y.to_numpy()
        [n,k] = X.shape

        Identity_mat = np.eye(k)
        self.w = np.matmul(np.linalg.inv(np.matmul(X.T, X) + self.lambda_val*Identity_mat), np.matmul(X.T, Y))
        
        #util.raiseNotDefined()
        return self.w