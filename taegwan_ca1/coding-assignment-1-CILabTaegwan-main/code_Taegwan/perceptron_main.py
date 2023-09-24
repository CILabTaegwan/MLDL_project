from numpy import *
import numpy as np
from util import *
import datasets
import logistic
import perceptron
from sklearn.linear_model import Perceptron

if __name__ == '__main__':
    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_cls()
    model = perceptron.Perceptron()
    thresh = 0.001
    model.setThreshold(thresh)
    eta=1/10000
    model.setEta(eta)
    a,b=model.train(tr_x, tr_y)
    y_hat = model.predict(val_x)
    val_y = np.array(val_y)
    acc = computeClassificationAcc(val_y, y_hat)
    print("accuracy of perceptron algorithm", format(acc))

    acc_history = [] #perceptron threshold 그래프 그리기용
    thresh_range = np.arange(0, 1, 0.001)
    for thr in thresh_range:
        model = perceptron.Perceptron()
        model.setThreshold(thr)
        eta = 1 / 10000
        model.setEta(eta)
        model.train(tr_x, tr_y)
        y_hat = model.predict(val_x)
        val_y = np.array(val_y)
        acc_history.append(computeClassificationAcc(val_y, y_hat))
    plt.clf()
    plt.plot(thresh_range, acc_history)
    plt.show()

    [test_x, test_y] = mydataset.testing_cls() #decision boundary 그리기용
    plt.clf()
    plt_list = list()
    plt_list2 = list()
    for i in range(569):
        plt_list.append(test_x[i][0])
        plt_list2.append(test_x[i][1])
    for i in range(569):
        if test_y[i] == 1:
            plt.plot(plt_list[i], plt_list2[i], 'gX')
        else:
            plt.plot(plt_list[i], plt_list2[i], 'mD')
    y = list()
    slope = -(b/a[1])/(b/a[0])
    intercept = -b/a[1]
    for i in range(569):
        y.append(slope*plt_list[i]+intercept)
    plt.plot(plt_list, y, 'r:')
    plt.show()

    new_model = Perceptron(penalty='l2',tol=1e-3, max_iter=1000) # sklearn perceptron
    new_model.fit(tr_x, tr_y)
    y_pred = new_model.predict(val_x)
    val_y = np.array(val_y)
    acc4 = computeClassificationAcc(val_y, y_pred)
    print("accuracy of perceptron with scikit learn model", format(acc4))

    
