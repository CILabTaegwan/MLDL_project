from numpy import *
import util
import pandas as pd
import numpy as np
class PriceDataset:
    """
    X is a feature vector
    Y is the predictor variable
    """
    tr_x = None  # X (data) of training set.
    tr_y = None  # Y (label) of training set.
    val_x = None # X (data) of validation set.
    val_y = None # Y (label) of validation set.

    def __init__(self):
        ## read the csv for training (price_data_tr.csv), 
        #                   val (price_data_val.csv)
        #                   and testing set (price_data_ts.csv)
        #
        ## CAUTION: the first row is the header 
        ##          (there is an option to skip the header 
        ##            when you read csv in python csv lib.)
        
        ### TODO: YOUR CODE HERE
        price_data_tr=pd.read_csv('../dataset/price_data_tr.csv')
        price_data_tr_x=pd.read_csv('../dataset/price_data_tr.csv')
        price_tr_x_feature = price_data_tr_x.drop(['id', 'date', 'price'], axis=1)
        
        price_data_val=pd.read_csv('../dataset/price_data_val.csv')
        price_data_val_x=pd.read_csv('../dataset/price_data_val.csv')
        price_val_x_feature = price_data_val_x.drop(['id', 'date', 'price'], axis=1)
        
        price_data_test_x=pd.read_csv('../dataset/price_data_ts.csv')
        price_data_test_x_list=price_data_test_x[['id','date']]
        price_data_test_x=price_data_test_x.drop(['id', 'date', 'price'], axis=1)
        
        self.tr_x =price_tr_x_feature.astype(float)
        self.tr_y = price_data_tr['price'].astype(float)
        self.val_x = price_val_x_feature.astype(float)
        self.val_y = price_data_val['price'].astype(float)

        self.ts_x = price_data_test_x.astype(float)
        self.ts_x_list = price_data_test_x_list


    def getDataset(self):
        return [self.tr_x, self.tr_y, self.val_x, self.val_y]
    
    def getTestDataset(self):
        return [self.ts_x, self.ts_x_list]
        
if __name__ == '__main__':
    price_dataset = PriceDataset()
    [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
    print(tr_x.shape)
    