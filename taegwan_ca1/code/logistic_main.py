from numpy import *
import numpy as np
import datasets
from binary import *
from util import *
from regression import *
import logistic
import sklearn.linear_model as sklearn
from sklearn.multiclass import OneVsRestClassifier
np.set_printoptions(threshold=sys.maxsize)


if __name__ == '__main__':

    price_dataset = datasets.PriceDataset()
    [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()

    quantize_levels = 100000
    tr_y_qtzd = util.quantizeY(tr_y, quantize_levels)
    val_y_qtzd = util.quantizeY(val_y, quantize_levels)
    y_classes = np.unique(tr_y_qtzd)
    num_classes = len(y_classes)
    prob_arr = []
    
    for i in range(num_classes):
        print(i)
        model = logistic.Logistic(n_iter=1000, lr=0.3)
        class_y = (tr_y_qtzd == y_classes[i]).astype(np.int64)
        model.train(tr_x, class_y)
        prob_arr.append(model.probability(val_x))
    y_hat = np.argmax(np.array(prob_arr), axis=0)
    y_hat_org = y_classes[y_hat] * quantize_levels
    error = computeAvgRegrMSError(y_hat_org, val_y)
    # My result
    print("computeAvgRegrMSError of logistic regression (My result): ", error)
    
    # sklearn result
    multiclass_logistic_regression = OneVsRestClassifier(sklearn.LogisticRegression(solver = 'lbfgs',max_iter = 1000, multi_class='multinomial'))
    model = multiclass_logistic_regression.fit(tr_x, tr_y_qtzd)
    y_hat = model.predict(val_x)
    error = computeAvgRegrMSError(y_hat, val_y)
    print("computeAvgRegrMSError of logistic regression (sklearn result): ", error)
    
    