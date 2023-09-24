from numpy import *
import numpy as np

from util import *

import datasets
import linear
import sklearn.linear_model as sklearn
import matplotlib.pyplot as plt


if __name__ == '__main__':
    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_reg()

    model = linear.Linear()
    lambda_val = 0.3
    model.setLam(lambda_val)
    a=model.train_ridge_CFS(tr_x, tr_y)
    y_hat = model.predict(val_x)
    error = computeAvgRegrMSError(val_y, y_hat)
    print("erorr of Ridge Regression Model with closed form solution:", format(error))


    #error_history = [] # Ridge regression(CFS) lambda 그래프 그리기
    #lambda_range = np.arange(0, 5.0, 0.5)
    #for lambda_val in lambda_range:
        #model =  linear.Linear()
        #model.setLam(lambda_val)
        #a=model.train_ridge_CFS(tr_x, tr_y)
        #y_hat = model.predict(val_x)
        #error_history.append(computeAvgRegrMSError(val_y, y_hat))
    #plt.plot(lambda_range, error_history)
    #plt.show()


    model = linear.Linear()
    lambda_val = 0.5
    eta=1/10000000
    model.setLam(lambda_val)
    model.setEta(eta)
    a=model.train_ridge_GD(tr_x, tr_y,1000)
    y_hat = model.predict_gd(val_x)
    error2 = computeAvgRegrMSError(val_y, y_hat)
    print("erorr of Ridge Regression Model with Gradient descent algorithm:", format(error2))

    #error_history = [] # Ridge regression(GD) eta 그래프 그리기
    #eta_range = np.arange(1 / 10000000, 1 / 1000000, 1 / 10000000)
    #for eta in eta_range:
     #   model = linear.Linear()
      #  model.setLam(0.5)
        #model.setEta(eta)
        #model.train_ridge_GD(tr_x, tr_y, 1000)
        #y_hat = model.predict_gd(val_x)
        #error_history.append(computeAvgRegrMSError(val_y, y_hat))
    #plt.plot(eta_range, error_history)
    #plt.show()

    new_model = sklearn.Ridge(fit_intercept = False, alpha=0.5)
    new_model.fit(tr_x,tr_y)
    y_hat = new_model.predict(val_x)
    error3 = computeAvgRegrMSError(val_y, y_hat)
    print("erorr of Ridge Regression Model with scikit learn model:", format(error3))


   # [test_x, test_y] = mydataset.testing_reg() # regressed line 그리기
   # plt_list = list()
   # plt_list2 = list()
   # for i in range(569):
    #    plt_list.append(test_x[i][0])
     #   plt_list2.append(test_x[i][1])
   # y = list()
    #for i in range(569):
     #   y.append(a[1] * plt_list[i] + a[0])
    #plt.clf()
    #plt.scatter(plt_list, test_y)
    #plt.plot(plt_list, y, 'r:')
    #plt.show()