import sys
import inspect
import random

from numpy import *
import numpy as np
from pylab import *
from matplotlib import pyplot
import util

#def visRegressedLine(X, predicted_y, w):
    ## TODO: YOUR CODE HERE
    
#def visClassifier(X, predicted_y, w):
    ### TODO: YOUR CODE HERE
    
#def visLoss(loss):
    ### TODO: YOUR CODE HERE
    
#def visLikelihood(likelihood):
    ### TODO: YOUR CODE HERE

def computeClassificationAcc(org_y, predicted_y):

    return np.sum(org_y == predicted_y)/org_y.size

def computeAvgRegrMSError(org_y, predicted_y):

    import numpy as np

    mse_loss=((org_y-predicted_y)**2).mean()
    mse_loss = np.sqrt(mse_loss)

    return mse_loss