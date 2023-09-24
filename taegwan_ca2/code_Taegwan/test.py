
from numpy import *
import numpy as np
import cv2
from kmeans import *
import kmeans
import util
import gmm
import matplotlib.pyplot as plt
### run and visualize your k-means algorithm 
def K_means(k,filename):
    model= kmeans.Kmeans(k,filename)
    model.run()
    plot=util

    plot.visDataCluster(model.error_list,k,model.X,model.Y)
    X=model.Y.tolist()



### run and visualize your GMM clustering 
def GMM(k, filename):
    gmm_model= gmm.GMM(k,filename)
    mu,cov, prob=gmm_model.run()

    category = prob.argmax(axis=1).flatten().tolist()[0]
    category=np.array(category)

    plot1 = util
    plot1.visLogLikelihood(mu, cov, prob,gmm_model.loglike,k, gmm_model.X, category)
def segementation(K):
    base_img = cv2.imread("tiger.jpg")
    img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    vectored_image = img.reshape((-1, 3))
    vectored_image = np.float32(vectored_image)
    criteria_eps = cv2.TERM_CRITERIA_EPS
    max_iter = cv2.TERM_CRITERIA_MAX_ITER
    rand_center= cv2.KMEANS_RANDOM_CENTERS
    criteria = (criteria_eps+max_iter, 100, 0.2)
    _, category, (point) = cv2.kmeans(vectored_image, K, None, criteria, 10, rand_center)
    point = np.uint8(point)
    category = category.flatten()
    output_image = point[category.flatten()]
    output_image = output_image.reshape(img.shape)
    # show the image
    plt.imshow(output_image)
    plt.show()



if __name__ == '__main__':
    K_means(3,"data2.npy")# 원하는 cluster 수와 데이터셋에 대한 k_means clustering 결과 생성
    GMM(3,"data0.npy") # 원하는 cluster 수와 데이터셋에 대한 em clustering 결과 생성
    segementation(10) #원하는 cluster 수에 대한 k-means clustering segnment image 생성