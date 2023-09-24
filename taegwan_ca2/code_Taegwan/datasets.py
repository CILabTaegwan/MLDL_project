from numpy import *
import numpy as np
#import util
import matplotlib.pyplot as plt
class Dataset:
    """
    X is a feature vector
    """
    def __init__(self, filename):
      
        self.data = np.load(filename)

    def getDataset_cluster(self):
        ### TODO: YOUR CODE HERE
        self.x = list()  ### TODO: YOUR CODE HERE
        for i in range(len(self.data[0])):

            self.x.append([self.data[0][i],self.data[1][i]])
        return self.x
    def get_rawdata(self):
        return self.data
