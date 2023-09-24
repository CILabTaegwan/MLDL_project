from numpy import *
import numpy as np
from sklearn.linear_model import LinearRegression
from binary import *
from util import *
from regression import *
import datasets
import linear

import csv

if __name__ == '__main__':

    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_reg()

    model = linear.Linear(0)
    a=model.train(tr_x, tr_y)
    y_hat = model.predict(val_x)
    error = computeAvgRegrMSError(val_y, y_hat) 
    print('\nMy linear regression result is {}\n'.format(error))
    
    
    lm = LinearRegression(fit_intercept=True, normalize=True, n_jobs=None)
    #lm = LinearRegression(fit_intercept=True, n_jobs=None)
    lm.fit(tr_x, tr_y)
    y_hat = lm.predict(val_x)
    error2 = computeAvgRegrMSError(val_y, y_hat) 
    print('The regression result of sklearn is {}\n'.format(error2))
    
    [ts_x, ts_list] = price_dataset.getTestDataset()
    y_hat = lm.predict(ts_x).astype(np.int64)
    f = open('submit.csv', 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(['id', 'price'])
    for i in range(len(y_hat)):
        wr.writerow(['{:010d}{}'.format(ts_list['id'][i], ts_list['date'][i]), y_hat[i]])
    f.close()