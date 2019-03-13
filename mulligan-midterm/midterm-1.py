from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import SpectralEmbedding

import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs

from numpy import genfromtxt

def pltPer(X, y, W):
    f = plt.figure()
    if X.shape[1] > 3:
        # reduce dimensions for display purposes
        Xw = np.append(X, np.reshape(W, (1, len(W))), 0)   # add a column of ones
        Xs = SpectralEmbedding(n_components=2).fit_transform(Xw)

        Xs = np.append(np.ones((Xs.shape[0],1)), Xs, 1)   # add a column of ones
        X = Xs[:-1,:]
        W = Xs[-1,:]
        
    for n in range(len(y)):
        if y[n] == -1:
            plt.plot(X[n,1],X[n,2],'r*')
        else:
            plt.plot(X[n,1],X[n,2],'b.')
    m, b = -W[1]/W[2], -W[0]/W[2]
    l = np.linspace(min(X[:,1]),max(X[:,1]))
    plt.plot(l, m*l+b, 'k-')
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.title("Perceptron Learning Algorithm")
    #plt.title("Linear Regression")

def classification_error(w, X, y):
    err_cnt = 0
    N = len(X)
    for n in range(N):
        s = np.sign(w.T.dot(X[n])) # if this is zero, then :(
        if y[n] != s:
            err_cnt += 1
    print(err_cnt)
    return err_cnt

def choose_miscl_point(w, X, y):
    mispts = []
    # Choose a random point among the misclassified
    for n in range(len(X)):
        if np.sign(w.T.dot(X[n])) != y[n]:
            mispts.append((X[n], y[n]))
    #print(len(mispts))
    return mispts[random.randrange(0,len(mispts))]

def main():
    itlst = []
    for x in range(1):

        # data
        
        dataset = genfromtxt('features.csv', delimiter=' ')
        y = dataset[:, 0]
        X = dataset[:, 1:]
        N = len(y)
        y[y!=4] = -1    #rest of numbers are negative class
        y[y==4] = +1    #number zero is the positive class
        X = np.append(np.ones((N,1)), X, 1)   # add a column of ones

        # Linear Regression
        #Xs = np.linalg.pinv(X.T.dot(X)).dot(X.T)
        #wlr = Xs.dot(y)

        #pltPer(X,y,wlr)
        #return

        # initialize the weigths to zeros
        w = np.zeros(3)
        #w = wlr
        it = 0
        pltPer(X,y,w)  # initial solution (bad!)
        
        stopIt = 10000
        currIt = 0
        bestW = w
        bestE = N
        # Iterate until all points are correctly classified
        nerr = classification_error(w, X, y)
        while nerr!= 0:
            it += 1
            currIt += 1
            if currIt > stopIt:
                print("Early stop! No progress!")
                break
            # Pick random misclassified point
            x, s = choose_miscl_point(w, X, y)
            # Update weights
            w += s*x
            nerr = classification_error(w, X, y)
            if nerr < bestE:
                currIt = 0
                bestE = nerr
                bestW = w

        w = bestW
        f = plt.figure()
        pltPer(X,y,w)
        print("Total iterations: " + str(it))
        itlst.append(it)
        #print(itlst)
    plt.title("Perceptron Learning Algorithm")
    #plt.title("Linear Regression")
    
main()