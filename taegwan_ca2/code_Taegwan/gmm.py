"""
A starting code for a Gaussian Mixture Model.
"""


import numpy as np
import datasets
from scipy.stats import multivariate_normal
class GMM:
    """
    This class is for the GMM implementation.
    """

    def __init__(self, K, filename):
        """
        Initialize our internal state.
        """
        self.types = 0
        if type(filename)==str:
            a = datasets.Dataset(filename)
            self.X = a.getDataset_cluster()
            self.X = np.array(self.X)

        else:
            self.X = filename
            self.types = 1
        self.Y = np.zeros(len(self.X))
        self.K = K
        self.N, self.D = self.X.shape

    def run(self):
        """
        Perform clustering
        """

        mu = np.random.rand(self.K, self.D)
        cov = np.array([np.eye(self.D)] * self.K)
        weight = np.array([1.0 / self.K] * self.K)
        prob=np.mat(np.zeros((self.N,self.K)))
        self.loglike = list()

        ### E-step
        ### TODO: YOUR CODE HERE
        for l in range(10):
            tmp_prob = np.zeros((self.N, self.K))
            for k in range(self.K):
                for n in range(self.N):
                    tmp_prob[n,k]= self.gaussian(self.X[n],mu[k],cov[k])

            tmp_prob=np.mat(tmp_prob)
            if self.types == 0:
                loglikevalue = self.log_likelihood(tmp_prob)
                self.loglike.append(loglikevalue)
                if l>1:
                    if self.loglike[l]==self.loglike[l-1]:
                        break
            for k in range(self.K):

                prob[:,k] = weight[k]*tmp_prob[:,k]

            for i in range(self.N):
                prob[i, :]/= np.sum(prob[i,:])
        ### M-step

            tmp_cov = list()

            for k in range(self.K):
                N_k = np.sum (prob[:,k])
                mu[k,:] = np.sum(np.multiply(self.X,prob[:,k]),axis=0)/N_k
                cov_k=(self.X-mu[k]).T*np.multiply((self.X-mu[k]),prob[:,k])/N_k
                tmp_cov.append(cov_k)
                weight[k]=N_k/self.N

            cov = np.array(tmp_cov)

        return mu, cov, prob
        ### TODO: YOUR CODE HERE

    #def stopping_criteria(self, loglike, prev_loglike):
    def stopping_criteria(self):
        """
        Compute convergence criteria
        """

    def log_likelihood(self, prob):

        tmp_value = np.zeros((self.N, 1))
        out_value=0
        for i in range(self.N):
            tmp_value[i, 0] = np.argmax(prob[i, :])
        temp =tmp_value.astype(int)
        for i in range(self.N):
            out_value += np.log(prob[i, temp[i, 0]])
        return out_value
    def gaussian(self, X, mu, sig):
        x_mu=X-mu
        sig_term =(2*np.pi)**self.D*np.linalg.det(sig)
        sig_term=1/np.sqrt(sig_term)
        exp_term= np.exp(-1*0.5*np.matmul(np.matmul(x_mu,np.linalg.inv(sig)),x_mu.T))
        return sig_term*exp_term




                
