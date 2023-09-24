import sys
import inspect
import random

from numpy import *
from pylab import *
from matplotlib import pyplot
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import util

def visDataCluster(error_list,k,X,Y):
    plt.clf()

    plt.plot(error_list)
    plt.xlabel("iteration")
    plt.ylabel("error function")
    plt.show()
    plt.clf()
    for idx in range (k):
        plt.scatter(X[Y==idx,0] , X[Y==idx,1], label = idx)
    plt.show()

    

def visLogLikelihood(mu, cov ,prob,error_list,k,X,Y):
    plt.clf()
    plt.plot(error_list)
    plt.xlabel("iteration")
    plt.ylabel("error function")
    plt.show()
    plt.clf()
    w_value = 0.2 / prob.max()
    for pos, covariance, w in zip(mu, cov, prob):
        plot_ellipse(pos, covariance, alpha=w_value)
    for idx in range(k):
        plt.scatter(X[Y == idx, 0], X[Y == idx, 1], label=idx)
    plt.show()


def plot_ellipse(pos, covariance, ax=None, **kwargs):

    ax = plt.gca() or ax

    # 분산을 주축으로 변환
    if covariance.shape == (2, 2):
        U, eigen, V = np.linalg.svd(covariance) #svd를 통해 3개의 변환값 저장
        w, h = 2 * np.sqrt(eigen)  # weight and height
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))

    else:
        w, h = 2 * np.sqrt(covariance)
        angle = 0

    for i in range(1, 4):
        ax.add_patch(Ellipse(pos, i * w, i * h, angle, **kwargs))
