
from util import *
import datasets
import logistic

from sklearn.linear_model import LogisticRegression
if __name__ == '__main__':
    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_cls()

    model = logistic.Logistic()
    lam = 3
    model.setLam(lam)
    eta = 1 / 10000
    model.setEta(eta)
    iter = 1000
    model.setMaxiter(iter)
    a, b = model.train_reg_SGA(tr_x, tr_y)
    y_hat = model.predict(val_x)
    val_y = np.array(val_y)
    acc3 = computeClassificationAcc(val_y, y_hat)

    print("accuracy of Regularized logistic regression model with Stochastic gradient ascent algorithm", format(acc3))

    acc_history = [] # Regularized Logistic Regression (SGA) lambda 그래프 그리기용
    lam_range = np.arange(0, 5, 0.1)
    for lam in lam_range:
        model = logistic.Logistic()
        model.setLam(lam)
        eta = 1 / 10000
        model.setEta(eta)
        iter = 1000
        model.setMaxiter(iter)
        a, b = model.train_reg_SGA(tr_x, tr_y)
        y_hat = model.predict(val_x)
        val_y = np.array(val_y)
        acc_history.append(computeClassificationAcc(val_y, y_hat))
    plt.clf()
    plt.plot(lam_range, acc_history)
    plt.show()

    [test_x, test_y] = mydataset.testing_cls()
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
    for i in range(569):
        y.append((a[0] * plt_list[i] + b)/(-a[1]))
    plt.plot(plt_list, y, 'r:')
    plt.show()
