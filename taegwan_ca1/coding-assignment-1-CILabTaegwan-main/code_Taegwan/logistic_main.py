
from util import *
import datasets
import logistic

from sklearn.linear_model import LogisticRegression

def Gradient_ascent(tr_x, tr_y, val_x, val_y):
    model = logistic.Logistic()
    eta = 1 / 10000
    model.setEta(eta)
    a, b = model.train_GA(tr_x, tr_y)
    y_hat = model.predict(val_x)
    val_y = np.array(val_y)
    acc = computeClassificationAcc(val_y, y_hat)

    print("accuracy of vanilla logistic regression model with gradient ascent algorithm", format(acc))

    acc_history = []
    eta_range = np.arange(0.01, 1 / 1000, 1 / 10000) # Logistic (GA) eta 그래프 그리기용
    for eta in eta_range:
        model_eta = logistic.Logistic()
        model_eta.setEta(eta)
        model_eta.train_GA(tr_x, tr_y)
        y_hat = model_eta.predict(val_x)
        acc_history.append(computeClassificationAcc(val_y, y_hat))
    plt.clf()
    plt.plot(eta_range, acc_history)
    plt.show()

    [test_x, test_y] = mydataset.testing_cls() # decision boundary 그리기용
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
        y.append((a[0] * plt_list[i] + b) / (-a[1]))
    plt.plot(plt_list, y, 'r:')
    plt.show()

def Stochastic_gradient(tr_x, tr_y, val_x, val_y):
    model = logistic.Logistic()
    eta = 1 / 10000
    model.setEta(eta)
    iter = 1000
    model.setMaxiter(iter)
    a, b = model.train_SGA(tr_x, tr_y)
    y_hat = model.predict(val_x)
    val_y = np.array(val_y)
    acc2 = computeClassificationAcc(val_y, y_hat)
    print("accuracy of vanilla logistic regression model with Stochastic gradient ascent algorithm", format(acc2))

    acc_history = [] # Logistic (SGA) iteration 그래프 그리기용
    itr_range = np.arange(0, 2000, 100)
    for itr in itr_range:
        model_itr = logistic.Logistic()
        model_itr.setMaxiter(itr)
        model_itr.train_SGA(tr_x, tr_y)
        y_hat = model_itr.predict(val_x)
        acc_history.append(computeClassificationAcc(val_y, y_hat))
    plt.clf()
    plt.plot(itr_range, acc_history)
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
    for i in range(569):
        y.append((a[0] * plt_list[i] + b) / (-a[1]))
    plt.plot(plt_list, y, 'r:')
    plt.show()




def Regularized_sga(tr_x, tr_y, val_x, val_y):
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

    print("accuracy of Regularized logistic regression model with Stochastic gradient ascent algorithm",
          format(acc3))

    acc_history = [] # Regularized Logistic Regression (SGA) lambda 그래프 그리기용
    lam_range = np.arange(0, 5, 0.1)
    for lam in lam_range:
        model_lam = logistic.Logistic()
        model_lam.setLam(lam)
        eta = 1 / 10000
        model_lam.setEta(eta)
        iter = 1000
        model_lam.setMaxiter(iter)
        #a, b = model_lam.train_reg_SGA(tr_x, tr_y)
        y_hat = model_lam.predict(val_x)
        val_y = np.array(val_y)
        acc_history.append(computeClassificationAcc(val_y, y_hat))
    plt.clf()
    plt.plot(lam_range, acc_history)
    plt.show()

    [test_x, test_y] = mydataset.testing_cls() # decision boundary 그리기용
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

def sklearn_reg(tr_x, tr_y, val_x, val_y):
    new_model = LogisticRegression(penalty='none',max_iter=500) #기존 Logisitc Regression과 비교
    new_model.fit(tr_x, tr_y)
    y_pred = new_model.predict(val_x)
    val_y = np.array(val_y)
    acc4 = computeClassificationAcc(val_y, y_pred)
    print("accuracy of vanilla logistic regression model with scikit learn model",format(acc4))


def sklearn_reg_l2(tr_x, tr_y, val_x, val_y):
    new_model = LogisticRegression(penalty='l2',max_iter=500) #기존 Logisitc Regression과 비교
    new_model.fit(tr_x, tr_y)
    y_pred = new_model.predict(val_x)
    val_y = np.array(val_y)
    acc5 = computeClassificationAcc(val_y, y_pred)
    print("accuracy of regularized logistic regression model with scikit learn model:",format(acc5))

if __name__ == '__main__':
    # 각 주석처리된 코드는 parameter (eta, iteartion lambda) 그래프 그리기용과 decision boundary 그리기용입니다
    mydataset = datasets.BreastCancerDataset()
    [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_cls()
    #Gradient_ascent(tr_x, tr_y, val_x, val_y)
    Stochastic_gradient(tr_x, tr_y, val_x, val_y)
    #Regularized_sga(tr_x, tr_y, val_x, val_y)
    #sklearn_reg(tr_x, tr_y, val_x, val_y)