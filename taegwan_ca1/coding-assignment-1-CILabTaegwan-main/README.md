[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-c66648af7eb3fe8bc4f294546bfd86ef473780cde1dea487d3c4ff354943c9ae.svg)](https://classroom.github.com/online_ide?assignment_repo_id=7800829&assignment_repo_type=AssignmentRepo)
# Coding Assignment 1: Linear/Ridge/Logistic regression models and Perceptron to diagnose breast cancer.

## Introduction

The first coding assignment asks you to implement three regression models and one classification model to diagnose breast cancer given features.

The assignment has three parts as following:
1. Implement the linear regression models and classification model; vanilla linear regression, ridge regression, logistic regression, regularized logistic regression and perceptron.
2. Compare your implementation with the same four models implemented in `scikit-learn`

We provide the code consisting of several Python files, which you will need to read and understand in order to complete the assignment.

**Note**: we will use `Python 3.x` for the project. 

## Deadline
May 10, 2022 11:59PM KST (*One day delay is permitted with linear scale score deduction.*)

### Submission checklist
* Push your code to [our github classroom page's CA1 section](https://classroom.github.com/a/MZou3O_s)
* Submit your report to [GEL](https://gel.gist.ac.kr)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

---
## Files

**Files you'll edit:**

* `datasets.py`: Data provider. 
* `linear.py`: You need to modify this file to implement a vanilla linear regression model.
* `logistic.py`: You need to modify this file to implement a logistic regression model.
* `util.py`: A bunch of utility functions!

---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

---
## Prepare the dataset (10%)

Load the dataset and divide this into train/val sets. 

```
>>> import datasets
>>> mydataset = datasets.BreastCancerDataset()
>>> [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_cls()
```

**For Linear and Ridge regressions**, use the last feature as the target values `y`, and the rest are used for `x`. 

```
>>> import datasets
>>> mydataset = datasets.BreastCancerDataset()
>>> [tr_x, tr_y, val_x, val_y] = mydataset.getDataset_reg()
```


---
## Vanilla Linear Regression Model (Closed form solution) (10%)

You can now implement the linear regression model to predict the feature (`y`) with other data (`x`).

```
>>> from linear import *
>>> model = Linear()
>>> model.train_CFS(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print(error)
0.241  # for example
```

`REPORT1`: Report the error and draw the regressed line using the first two features. Discuss any ideas to reduce the errors (e.g., new feature transforms or using kernels or etc.)

---
## Ridge Regression Model (Closed form solution) (10%)

Same as before but implement a ridge regression model. 

```
>>> from linear import *
>>> model = Linear()
>>> lambda = 1.0
>>> model.setLam(lambda)
>>> model.train_ridge_CFS(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print(error)
0.1542  # for example
```

`REPORT2`: Report the error and draw the regressed line using the first two features. Sweep `lambda` from 0.0 to 5.0 (or some other reasonable values) with a reasonable sized step (e.g., 0.5), plot a graph (x-axis: lambda, y-axis: error) and discuss the effect of the lambda (especially comparing with vanilla linear when `lambda=0`.)

---
## Ridge Regression Model (Gradient descent algorithm) (10%)

Same as before but implement a ridge regression model. 

```
>>> from linear import *
>>> model = Linear()
>>> lambda = 1.0
>>> model.setLam(lambda)
>>> eta = 1.0
>>> model.setEta(eta)
>>> model.train_ridge_GD(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print(error)
0.1542  # for example
```

`REPORT3`: Report the error and draw the regressed line using the first two features. Sweep `eta` from 0.01 to 10.0 (or some other reasonable values) with a reasonable sized step (0.01, 0.1, etc), plot a graph (x-axis: eta, y-axis: error) and discuss the effect of the eta.

---
## Logistic Regression Model (Gradient ascent algorithm) (10%)

You can now implement the logistic regression model to predict the class (`y`) with the input data (`x`).

```
>>> from logistic import *
>>> model = Logistic()
>>> eta = 1.0
>>> model.setEta(eta)
>>> model.train_GA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT4`: Report the error of your logistic regression model and draw the decision boundary using the first two features. Sweep `eta` from 0.01 to 10.0 (or some other reasonable values) with a reasonable sized step (0.01, 0.1, etc), plot a graph (x-axis: eta, y-axis: acc) and discuss the effect of the eta.

---
## Logistic Regression Model (Stochastic Gradient ascent algorithm) (10%)

You can now implement the logistic regression model to predict the class (`y`) with the input data (`x`).

```
>>> from logistic import *
>>> model = Logistic()
>>> eta = 1.0
>>> model.setEta(eta)
>>> iter = 1000
>>> model.setMaxiter(iter)
>>> model.train_SGA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT5`: Report the error of your logistic regression model and draw the decision boundary using the first two features. Using the `eta` you found from `REPORT4`, run with different numbers of `iterations`, plot a graph (x-axis: eta, y-axis: acc) and discuss the effect of the number of iterations.

---
## Regularized Logistic Regression Model (Stochastic Gradient ascent algorithm) (10%)

You can now implement the regularized logistic regression model to predict the class (`y`) with the input data (`x`).


```
>>> from logistic import *
>>> model = Logistic()
>>> lambda = 1.0
>>> model.setLam(lambda)
>>> eta = 1.0
>>> model.setEta(eta)
>>> iter = 1000
>>> model.setMaxiter(iter)
>>> model.train_reg_SGA(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT6`: Report the error of your regularized logistic regression model and draw the decision boundary using the first two features. Sweep `lambda` from 0.0 to 5.0 (or some other reasonable values) with a reasonable sized step (e.g., 0.5), plot a graph (x-axis: lambda, y-axis: acc) and discuss the effect of the lambda (especially comparing with vanilla logistic when `lambda=0`.)



---
## Perceptron (10%)

You can now implement the perceptron to predict the class (`y`) with the input data (`x`).


```
>>> from perceptron import *
>>> model = Perceptron()
>>> thresh = 0.001
>>> model.setThreshold(thresh)
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> acc = computeClassificationAcc(val_y, y_hat) 
>>> print(acc)
0.96  # for example
```

`REPORT7`: Report the error of your perceptron and draw the decision boundary using the first two features. Using different values for `threshold`, plot a graph (x-axis: thresh, y-axis: acc) and discuss the effect of the threshold. Discuss what happens when 'threshold=0' and why.



---
## Compare your implementations with `scikit-learn` library (20%)

In [scikit-learn library](https://scikit-learn.org/), there are all implementation of what you have implemented
1. [vanilla linear regressor](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
2. [ridge regressor](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
3. [vanilla logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
4. [regularized logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)
5. [perceptron](https://scikit-learn.org/stable/modules/linear_model.html#perceptron)

`REPORT8`: Compare the error by your implementations of vanilla linear regression and OLS model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT9`: Compare the error by your implementations of ridge regression and ridge regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT10`: Compare the error by your implementations of logistic regression and logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT11`: Compare the error by your implementations of l2-regularized logistic regression and l2-regularized logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT12`: Compare the error by your implementations of perceptron and perceptron model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

