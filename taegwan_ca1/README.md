# Coding Assignment 1: Linear regression models to estimate housing price.

## Introduction

The first coding assignment asks you to implement three regression models and one classification model to predict house price given features.

The assignment has three parts as following:
1. Implement the three linear regression models and one classification model; vanilla linear regression, ridge regression, LASSO and logistic regression
1. Compare your implementation with the same four models implemented in `scikit-learn`
3. Improve your accuracy and compete with your colleagues in [kaggle page of this coding assignment](https://www.kaggle.com/t/b586c10fd3af410e99203656ec90a4c0)

We provide the code consisting of several Python files, some of which you will need to read and understand in order to complete the assignment, and some of which you can ignore. You will figure them out which one is which.

You can use some deep learning libraries (e.g., PyTorch, Tensorflow) to accelerate your code with CUDA back-end.

**Note**: we will use `Python 3.x` for the project.

## Deadline
November 9, 2020 11:59PM KST (*One day delay is permitted with linear scale score deduction.*)

### Submission checklist
* Push your code to [our github classroom page's CA1 section](https://classroom.github.com/a/cI0rINhC)
* Submit your report to [Gradescope 'CA1 Report' section](https://www.gradescope.com/courses/158113)
* Submit your entry to [Kaggle leaderboard](https://www.kaggle.com/c/gistpriceest2020/leaderboard)

---
## Preparation

### Installing prerequisites

The prerequisite usually refers to the necessary library that your code can run with. They are also known as `dependency`. To install the prerequisite, simply type in the shell prompt (not in a python interpreter) the following:

```
$ pip install -r requirements.txt
```

### Download the dataset

Go to [dataset page in our kaggle page for this challenge](https://www.kaggle.com/c/gistpriceest2020/data) to download the dataset. Copy (or move) the dataset into `./dataset` sub-directory.

---
## Files

**Files you'll edit:**

* `datasets.py`: Data provider. 
  - Implement functions to read the `./dataset/price_data_tr.csv` and `./dataset/price_data_val.csv` file. 
    - The recommended interface has been noted in the file as comments.
* `linear.py`: You need to modify this file to implement a vanilla linear regression model.
* `ridge.py`: You need to modify this file to implement a ridge regression model.
* `logistic.py`: You need to modify this file to implement a logistic regression model.

**Files you may want to review:**

* `binary.py`: A generic interface for binary classifiers or regressions.
* `regression.py`: An abstract class for the regression models.
* `util.py`: A bunch of utility functions!
* `mlGraphics.py`: Plotting commands.
* `dumbClassifiers.py`: Implementations of very simple classifiers. You may refer to them for your reference to implement other classifiers.

---
## What to submit
**Push to your github classroom** 

- All of the python files listed above (under "Files you'll edit"). 
  - **Caution:** DO NOT UPLOAD THE DATASET
- `report.pdf` file that answers all the written questions in this assignment (denoted by `"REPORT#:"` in this documentation).
  - If you do not want to use LaTeX, use any other wordprocessor and render it to PDF.

---
### Note
**Academic dishonesty:** We will be checking your code against other submissions in the class for logical redundancy. If you copy someone else's code and submit it with minor changes, we will know. These cheat detectors are quite hard to fool, so please don't try. We trust you all to submit your own work only; please don't let us down. If you do, we will pursue the strongest consequences available to us.

**Getting help:** You are not alone! If you find yourself stuck on something, contact the course staff for help. Office hours, class time, and Piazza are there for your support; please use them. We want these projects to be rewarding and instructional, not frustrating and demoralizing. But, we don't know when or how to help unless you ask.

---
## Prepare the dataset (5%)

Read the csv to load the dataset.

```
>>> import datasets
>>> price_dataset = datasets.PriceDataset()
>>> [tr_x, tr_y, val_x, val_y] = price_dataset.getDataset()
```

You may ignore the warning of `Fontconfig warning: ignoring UTF-8: not a valid region tag` after `import datasets` command.

---

## Vanilla Linear Regression Model (10%)

You can now implement the linear regression model to predict the price (`y`) with other data (`x`).

```
>>> import linear
>>> model = Linear()
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.241  # for example
```

`REPORT1`: Report the error. Discuss any ideas to reduce the errors (e.g., new feature transforms or using kernels or etc.)

---
## Ridge Regression Model (10%)

Same as before but implement a ridge regression model. 

```
>>> import ridge
>>> model = Ridge()
>>> lambda = 1.0
>>> model.setLambda(lambda)
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.1542  # for example
```

`REPORT2`: Sweep `lambda` from 0.0 to 5.0 (or some other reasonable values) with a reasonable sized step (e.g., 0.5), plot a graph (x-axis: gamma, y-axis: accuracy) and discuss the effect of the gamma (especially comparing with vanilla linear when `lambda=0`.)


---
## Multi-class Logistic Regression (Optimized by Stochastic Gradient Descent) (25%)

In class, we learn the logistic regression model is for binary classification. Here, we are solving a regression problem (although the name of the classifier seems for regression). To change the regression problem into classification problem, we can quantize the price by **100,000 steps (then you'll have 77 classes)**. The regression precision would be bounded by 100,000 but you can solve the problem. Or you can encode the steps more fine-grained or varied step sized (eg., In 75,000 to 1,000,000, quantize it by 50K and more than 1M as a single class).

In class, we learn how to implement binary class logistic regression model and simple extension to multi-class problem. Here, we ask you to implement multi-class Logistic regression model. There are two ways of implementing multi-class logistic regression model. 1) a one-versus-all binary logistic regression models, 2) changing logistic loss to cross entropy loss [a useful page](https://peterroelants.github.io/posts/cross-entropy-logistic/).

One idea to improve accuracy when you use logistic regression model for 77 classes is to combine logistic regression to obtain rough range (within one class) and regress further with other regression models (e.g., vanilla linear, ridge or LASSO). If you combine logistic regression with other regression models, training the secondary regression model needs a thought as regressing within each range should be different (you may further assume to regress within a range should act similarly regardless which range you're regressing on).

```
>>> import logistic  # binary logistic classifier
>>> import numpy as np
>>> models = []
>>> tr_y_qtzd = quantizeY(tr_y)
>>> y_classes = np.unique(tr_y_qtzd)
>>> num_classes = len(y_classes)
>>> for i in range(num_classes):
...     model = Logistic()
...     model.train(tr_x, tr_y_qtzd)  # train with 'one-vs-all' manner
...     models.append(model)
>>> max_conf = 0
>>> max_class_idx = 0
>>> for i in range(num_classes):
...     _, confidence = models[i].predict(val_x)
...     if confidence > max_conf:
...         max_conf = confidence
...         max_class_idx = i
>>> val_y_hat = y_classes[max_class_idx]
>>> error = computeAvgRegrMSError(val_y_hat, y_hat) 
>>> print error
0.1387  # for example
```

`REPORT4`: Report the error of your logistic regression model. 

`REPORT5`: Discuss any idea of improvement (e.g., combine logistic regression with further regressor or add regularizers.).

---
## LASSO (Optimized by Feature-sign algorithm) (35%+Extra Credit)

LASSO has multiple solvers. Each of the solvers has trade-off between accuracy and efficiency. The last implementation task is to implement [feature-sign algorithm by H. Lee](https://web.eecs.umich.edu/~honglak/nips06-sparsecoding.pdf) for solving LASSO in Python. Refer to `Algorithm 1` in the linked paper. Also, there is [a Matlab implementation of the feature-sign algorithm](https://web.eecs.umich.edu/~honglak/softwares/fast_sc_r2.zip). You can refer to the Matlab implementation for implementing your version.

If you further improve your implementation, we will give you a plenty of extra credit.

```
>>> import lasso
>>> import numpy as np
>>> models = []
>>> lambda = 1.0
>>> model.setLambda(lambda)
>>> model.train(tr_x, tr_y)
>>> y_hat = model.predict(val_x)
>>> error = computeAvgRegrMSError(val_y, y_hat) 
>>> print error
0.0932  # for example
```

`REPORT6`: Report the error of your LASSO model by sweeping `lambda` with the same set of values you have tried for ridge regression, plot a graph (x-axis: gamma, y-axis: accuracy) and discuss the effect of the gamma (especially comparing with vanilla linear when `gamma=0`.)

`REPORT7`: Overlay graph of ridge regression and LASSO by the same `lambda`. Discuss the difference between vanilla, ridge and LASSO.

`REPORT8`: Discuss of your improvement if there are any.

---
## Compare your implementations with `scikit-learn` library (10%)

In [scikit-learn library](https://scikit-learn.org/), there are all implementation of what you have implemented
1. [vanilla linear regressor](https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares)
1. [ridge regressor](https://scikit-learn.org/stable/modules/linear_model.html#ridge-regression)
1. [logistic regression model](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression). 

Please make sure your implementations of vanilla linear regressor and ridge regressor are *exactly* the same to the ones in scikit-learn. Please compare outputs of your version of logistic regressions and the scikit-learn version.

`REPORT9`: Compare the error by your implementations of vanilla linear regression and OLS model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT10`: Compare the error by your implementations of ridge regression and ridge regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

`REPORT11`: Compare the error by your implementations of LASSO and LASSO model in scikit-learn and discuss the reason for the difference.

`REPORT12`: Compare the error by your implementations of logistic regression and logistic regression model in scikit-learn and discuss the reason for the difference. If they are identical, report and claim you're awesome!

---
## Compete with others on the accuracy using Kaggle (5%)

**Caution**
1. Use your github ID for your team name, otherwise we can't figure out who you are.
1. Do not over-engineer your method by tuning hyper-parameters heavily.

Using either your implementation of other regressors or tune the outputs of libraries implemented in scikit-learn, please get the best accuracy of price prediction and compete with other students in the Kaggle platform. We'll release the competition detailed soon in a Piazza posting.

If you use your implementation to win the competition, we will give you a big extra credit.

`REPORT13`: Discuss all trials you've done with either with your implementations or scikit-learn library's functions.

