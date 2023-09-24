"""
A starting code for a K-means algorithm.
"""

from numpy import *
import numpy as np
import datasets

class Kmeans:
    """
    This class is for the K-means implementation.
    """

    def __init__(self, K,filename):
        """
        Initialize our internal state.
        """
        ### TODO: YOUR CODE HERE
        if type(filename)==str:
            a=datasets.Dataset(filename)
            self.X=a.getDataset_cluster()
            self.X = np.array(self.X)
        else:
            self.X = filename
        self.Y=np.zeros(len(self.X))
        self.K=K

    def run(self):
        """
        Perform clustering
        """

        rnd_idx= np.random.choice(len(self.X), self.K, replace=False)
        self.centroids =list()
        self.past_centroids=list()
        for i in rnd_idx:
            self.centroids.append(self.X[i].tolist())

        self.error_list=list()
        while self.stopping_criteria()==0:
            ### Assignment step
            dist=[]
            self.past_centroids=self.centroids
            for i in range(len(self.X)):
                dist_for_K=[]
                for j in range(self.K):
                    dist_for_K.append(self.calc_dist(self.X[i], self.centroids[j]))
                dist.append(dist_for_K)
            distance= sum([np.min(i) for i in dist])
            self.Y = np.array([np.argmin(i) for i in dist])

            ### Update step
            self.centroids=list()
            error_dist = 0
            for idx in range(self.K):
                temp_cent=self.X[self.Y==idx].mean(axis=0)
                temp_cent=temp_cent.tolist()
                self.centroids.append(temp_cent)
                dist_for_error=list()
                for i in range(len(self.X[self.Y==idx])):
                    dist_for_error.append(self.calc_dist(self.X[self.Y==idx][i], temp_cent))# i 넣으면 안됨
                error_dist+=sum(dist_for_error)
            self.error_list.append(error_dist)



    def stopping_criteria(self):
        """
        Compute convergence criteria
        """
        a=(self.centroids==self.past_centroids)
        return a




    def calc_dist(self, X, Y):
        """
        Compute distance between two vectors
        """
        return (X[0]-Y[0])**2+(X[1]-Y[1])**2




