from numpy import *
import numpy as np
from binary import *
from util import *
from regression import *
import datasets
import ridge
import sklearn.linear_model as sklearn
import matplotlib.pyplot as plt

if __name__ == '__main__':

    price_dataset = datasets.PriceDataset()
    [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()

    model = ridge.Ridge(0)
    lambda_val = 1.0
    model.setLambda(lambda_val)
    model.train(tr_x, tr_y)
    y_hat = model.predict(val_x)
    error = computeAvgRegrMSError((val_y), (y_hat)) 
    print('\nMy ridge regression result is {}\n'.format(error))
    
    
    error = []
    lambda_range = np.arange(0, 5.0, 0.1)
    for _lambdaVal in lambda_range:
        model = ridge.Ridge(0)
        model.setLambda(_lambdaVal)
        model.train(tr_x, tr_y)
        y_hat = model.predict(val_x)
        error.append(computeAvgRegrMSError(val_y, y_hat))
    plt.plot(lambda_range, error)
    plt.show()
    plt.savefig('fig1.png', dpi=300)
    
    lin_reg = sklearn.Ridge(fit_intercept = False, alpha=1)
    lin_reg.fit(tr_x,tr_y)
    y_hat = lin_reg.predict(val_x)
    error2 = computeAvgRegrMSError(val_y, y_hat)
    print('The ridge regression of sklearn is: {}\n'.format(error2))