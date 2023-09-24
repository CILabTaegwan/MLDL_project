import sys
import inspect
import random

from numpy import *
from pylab import *

import util
import binary


def raiseNotDefined():
  print('Method not implemented: %s' % inspect.stack()[1][3])    
  sys.exit(1)

def permute(a):
  """
  Randomly permute the elements in array a
  """
  for n in range(len(a)):
    m = int(pylab.rand() * (len(a) - n)) + n
    t = a[m]
    a[m] = a[n]
    a[n] = t
    
def splitTrainTest(X0, Y0, freqTest):
  """
  Split data in X0/Y0 into train/test data with freqTest
  frequency of test points
  """
  N,D = X0.shape
  isTest = zeros(N, dtype=bool)
  for n in range(0, N, freqTest):
    isTest[n] = True
  X   = X0[isTest==False, :]
  Y   = Y0[isTest==False]
  Xte = X0[isTest, :]
  Yte = Y0[isTest]

  return (X,Y,Xte,Yte)


def uniq(seq, idfun=None): 
  # order preserving
  if idfun is None:
    def idfun(x): return x
  seen = {}
  result = []
  for item in seq:
    marker = idfun(item)
    # in old Python versions:
    # if seen.has_key(marker)
    # but in new ones:
    if marker in seen: continue
    seen[marker] = 1
    result.append(item)
  return result

def mode(seq):
  if len(seq) == 0:
    return 1.
  else:
    cnt = {}
    for item in seq:
      if cnt.has_key(item):
        cnt[item] += 1
      else:
        cnt[item] = 1
    maxItem = seq[0]
    for item,c in cnt.iteritems():
      if c > cnt[maxItem]:
        maxItem = item
    return maxItem



def plotCurve(titleString, res):
    plot(res[0], res[1], 'b-',
         res[0], res[2], 'r-')
    legend( ('Train', 'Test') )
    ylabel('Accuracy')
    title(titleString)
    show()

def shufflePoints(X, Y):
    """
    Randomize the order of the points.
    """

    [N,D] = X.shape
    order = range(N)
    util.permute(order)

    retX = X[order,:]
    retY = Y[order]
    return (retX, retY)
            

def plotData(X, Y):
    plot(X[Y>=0,0], X[Y>=0,1], 'bo',
         X[Y< 0,0], X[Y< 0,1], 'rx')
    legend( ('+1', '-1') )
    show()

def plotClassifier(w, b):
    axes = figure(1).get_axes()[0]
    xlim = axes.get_xlim()
    ylim = axes.get_ylim()

    xmin = xlim[0] + (xlim[1] - xlim[0]) / 100
    xmax = xlim[1] - (xlim[1] - xlim[0]) / 100
    ymin = ylim[0] + (ylim[1] - ylim[0]) / 100
    ymax = ylim[1] - (ylim[1] - ylim[0]) / 100

    # find the zeros along each axis
    # w0*l + w1*? + b = 0  ==>  ? = -(b + w0*l) / w1
    xmin_zero = - (b + w[0] * xmin) / w[1]
    xmax_zero = - (b + w[0] * xmax) / w[1]
    ymin_zero = - (b + w[1] * ymin) / w[0]
    ymax_zero = - (b + w[1] * ymax) / w[0]

    # now, two of these should actually be in bounds, figure out which
    inBounds = []
    if ylim[0] <= xmin_zero and xmin_zero <= ylim[1]:
        inBounds.append( (xmin, xmin_zero) )
    if ylim[0] <= xmax_zero and xmax_zero <= ylim[1]:
        inBounds.append( (xmax, xmax_zero) )
    if xlim[0] <= ymin_zero and ymin_zero <= xlim[1]:
        inBounds.append( (ymin_zero, ymin) )
    if xlim[0] <= ymax_zero and ymax_zero <= xlim[1]:
        inBounds.append( (ymax_zero, ymax) )

    plot( array([inBounds[0][0], inBounds[1][0]]), array([inBounds[0][1], inBounds[1][1]]), 'g-', linewidth=2 )
    figure(1).set_axes([axes])
    
def dumpMegamFormat(fname, Xtr, Ytr, Xte, Yte):
    def writeIt(f, X, Y):
        N,D = X.shape
        for n in range(N):
            f.write(str(Y[n]))
            for d in range(D):
                if X[n,d] != 0:
                    f.write(" f" + str(d) + " " + str(X[n,d]))
            f.write("\n")

    f = open(fname, 'w')
    writeIt(f, Xtr, Ytr)
    f.write("TEST\n")
    writeIt(f, Xte, Yte)
    f.close()

def dumpMegamFormatSet(fname, dataset):
    dumpMegamFormat(fname, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)

def dumpSVMFormat(fname, Xtr, Ytr, Xte, Yte):
    def writeIt(f, X, Y):
        N,D = X.shape
        for n in range(N):
            f.write(str(Y[n]))
            for d in range(D):
                if X[n,d] != 0:
                    f.write(" " + str(d+1) + ":" + str(X[n,d]))
            f.write("\n")

    f = open(fname, 'w')
    writeIt(f, Xtr, Ytr)
    writeIt(f, Xte, Yte)
    f.close()


def dumpSVMFormatSet(fname, dataset):
    dumpSVMFormat(fname, dataset.X, dataset.Y, dataset.Xte, dataset.Yte)


def quantizeY(org_y, quantize_levels):
    ### TODO: YOUR CODE HERE
    
    quantize_output = (org_y / quantize_levels).astype(np.int64)
    # raise NotImplementedError
    return quantize_output

def computeClassificationAcc(org_y, predicted_y):
    '''
        Compute classification accuracy by counting how many predicted_y
        is the same to the org_y
    '''
    ### TODO: YOUR CODE HERE
    return np.sum(org_y == predicted_y)/org_y.size

def computeAvgRegrMSError(org_y, predicted_y):
    '''
        Compute regression error by average error between predicted_y
        and org_y. Use L2 distance between two values (each eleement 
        in the vector).
    '''
    ### TODO: YOUR CODE HERE
    mse_loss = np.sqrt(((org_y-predicted_y)**2).mean())
    #raise NotImplementedError
    return mse_loss
