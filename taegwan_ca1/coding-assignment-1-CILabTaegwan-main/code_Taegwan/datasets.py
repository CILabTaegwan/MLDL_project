from numpy import *
import numpy as np
import matplotlib
from copy import deepcopy
from sklearn.datasets import load_breast_cancer
from random import randrange
class BreastCancerDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """

    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.

    def __init__(self):
      
        dataset = load_breast_cancer()
        self.data_x = dataset['data']
        self.data_y = dataset['target']



        self.tr_x = None
        self.tr_y = None
        self.val_x = None
        self.val_y = None

    def getDataset_cls(self):
        ### TODO: YOUR CODE HERE

        train_size=0.7*len(self.data_x)
        self.tr_x=list()
        self.tr_y = list()
        self.val_x= list(self.data_x)
        self.val_y=list(self.data_y)
        index=0
        while len(self.tr_x)<train_size:
            index= randrange(len(self.val_x))
            self.tr_x.append(self.val_x.pop(index))
            self.tr_y.append(self.val_y.pop(index))

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
    def testing_cls(self): #classfication 데이터 그리기용
        ### TODO: YOUR CODE HERE

        train_size=0.7*len(self.data_x)
        self.tr_x=list()
        self.tr_y = list()
        self.val_x= list(self.data_x)
        self.val_y=list(self.data_y)

        return [self.val_x, self.val_y]

    def getDataset_reg(self):
        ### TODO: YOUR CODE HERE

        self.tr_x = list()
        self.tr_y = list()
        self.val_x = list()
        self.val_y = list()
        train_size = 0.7 * len(self.data_x)

        for i in range(569):
            temp_data=list(self.data_x[i])
            self.val_y.append(temp_data.pop(29))
            self.val_x.append(temp_data)
        index=0
        while len(self.tr_x) < train_size:
            index = randrange(len(self.val_x))
            self.tr_x.append(self.val_x.pop(index))
            self.tr_y.append(self.val_y.pop(index))

        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
    def testing_reg(self): #regression 데이터 그리기용
        ### TODO: YOUR CODE HERE

        self.tr_x = list()
        self.tr_y = list()
        self.val_x = list()
        self.val_y = list()
        train_size = 0.7 * len(self.data_x)

        for i in range(569):
            temp_data=list(self.data_x[i])
            self.val_y.append(temp_data.pop(29))
            self.val_x.append(temp_data)


        return [self.val_x, self.val_y]

if __name__ == '__main__':
    print("test")
