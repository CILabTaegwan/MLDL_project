"""
A starting code for a LASSO linear regression model.  
This implementation should be based on the Feature-sign algorithm by H. Lee.
"""

from numpy import *

from binary import *
import util
from regression import *

class Lasso(Regression):
    """
    This class is for the ridge regressor implementation.
    """

    w = None
    lambda = 0.0

    def __init__(self, opts):
        """
        Initialize our internal state.  The options are:
          opts.maxDepth = maximum number of features to split on
                          (i.e., if maxDepth == 1, then we're a stump)
        """

        self.opts = opts

    def setLambda(self, lambdaVal):
        self.lambda = lambdaVal

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
        util.raiseNotDefined()

    def train(self, X, Y):
        """
        Build a LASSO regressor using Feature-Sign algorithm by H. Lee (NeurIPS 2007).
        """
        ### TODO: YOUR CODE HERE
        util.raiseNotDefined()