from numpy import *
import numpy as np
from sklearn.linear_model import LinearRegression
import datasets
from util import *
import linear
import numpy as np
import matplotlib.pyplot as plt
import csv
import linear

if __name__ == '__main__':

    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_reg()

    model = linear.Linear()
    a=model.train_CFS(tr_x, tr_y)
    y_pred = model.predict(val_x)
    error = computeAvgRegrMSError(val_y, y_pred)
    print("erorr of Vanilla Linear Regression Model:",format(error))
    [test_x,test_y] = mydataset.testing_reg() #regressed line 그리기용
    plt_list = list()
    plt_list2= list()
    for i in range(569):
        plt_list.append(test_x[i][0])
        plt_list2.append(test_x[i][1])
    plt.scatter(plt_list2,test_y)
    y=list()
    for i in range(569):
        y.append(a[2]*plt_list2[i]+a[0])
    plt.clf()
    plt.scatter(plt_list2, test_y)
    plt.plot(plt_list2,y,'r:')
    plt.show()




    new_model = LinearRegression(fit_intercept=True, n_jobs=None)
    new_model.fit(tr_x, tr_y)
    y_pred = new_model.predict(val_x)
    error2 = computeAvgRegrMSError(val_y, y_pred)
    print("error of Linear regression with scikit learn model:",format(error2))

    
