"""
A starting code for a vanilla linear regression model.
"""

from numpy import *
import numpy as np
from binary import *
import util
from regression import *
import datasets

class Linear(Regression):
    """
    This class is for the linear regression model implementation.  
    """
    
    w = None

    def __init__(self, opts):
        self.opts = opts  # options

    def online(self):
        ### TODO: YOU MAY MODIFY THIS
        return False

    def __repr__(self):
        """
        Return a string representation of the linear model
        """
        return self.w

    def __str__(self):
        """
        Return a string representation of the linear model
        """
        return self.w

    def predict(self, X):
        """
        Perform inference
        """
        ### TODO: YOUR CODE HERE
        X = X.to_numpy()
        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        predict = np.matmul(X, self.w)
        #util.raiseNotDefined()
        
        return predict

    def train(self, X, Y):
        """
        Build a vanilla linear regressor.
        """
        ### TODO: YOUR CODE HERE

        intercept = np.ones((X.shape[0], 1))
        X = np.concatenate((intercept, X), axis=1)
        self.w = np.linalg.pinv(X.T @ X) @ X.T @ Y
        #util.raiseNotDefined()
        return self.w
        
if __name__ == '__main__':
    '''
    price_dataset = datasets.PriceDataset()
    [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
    print(tr_x)
    model = Linear(0)
    model.train(tr_x, tr_y)
    '''
    