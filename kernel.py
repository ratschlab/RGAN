#!/usr/bin/env ipython
# Experimenting with kernel between two equal-length "time series".
# Note: kernel must be consistent for MMD formuluation to be valid.
# Note: the time series *may* be multi-dimensional.

import numpy as np
import scipy as sp
#from sklearn.metrics.pairwise import my_rbf
import matplotlib.pyplot as plt
import re
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import pdb

import data_utils

seq_length = 30
num_signals = 1
num_samples = 10
X, pdf = data_utils.GP(seq_length, num_samples, num_signals)
# for testing: make Y quite similar to X
#X = data_utils.sine_wave(seq_length, num_samples, num_signals)
#, freq_low=2, freq_high=2.1)
#X = np.random.normal(size=(num_samples, seq_length, num_signals), scale=0.5)
#for i in range(1, num_samples):
#    X[i] = X[i-1] + np.random.normal(size=(seq_length, num_signals), scale=0.3)

def cos_dist(x, y):
    dist = 0.5*(1 - np.dot(x.T,y)[0, 0]/(np.linalg.norm(x)*np.linalg.norm(y)))
    return dist

def my_rbf(x, y=None, gamma=1.0/(2.1)**2, withnorm=False):
    """
    """
    if y is None:
        y = x
    if withnorm:
        xn = x/np.linalg.norm(x)
        yn = y/np.linalg.norm(y)
    else:
        xn = x
        yn = y
    dist = np.linalg.norm(xn - yn)
    return np.exp(-gamma*(dist**2))

def compare_metrics(X, num=10):
    """
    """
    fig, axarr = plt.subplots(num, 4, figsize=(15, 15))
    xx = np.arange(30)
    fig.suptitle(' '.join(['dtw', 'cos', 'euc', 'rbf']))
    for (col, distance_measure) in enumerate([fastdtw, cos_dist, euclidean, my_rbf]):
        dists = []
        for i in range(num):
            try:
                d, _ = distance_measure(X[0], X[i])
            except TypeError:
                d = distance_measure(X[0], X[i])
            if col == 3:
                d = -d
            dists.append(dtw)
        # now, plot in order
        for (i, j) in enumerate(np.argsort(dists)):
            axarr[i, col].plot(xx, X[j])
            axarr[i, col].plot(xx, X[0], alpha=0.5)
            dtw, _ = fastdtw(X[0], X[j])
            title = '%.1f %.1f %.1f %.1f' % (dtw, cos_dist(X[0], X[j]), euclidean(X[0], X[j]), my_rbf(X[0], X[j]))
            #title = '%.1f' % (dtw)
            axarr[i, col].set_title(title)
            axarr[i, col].set_ylim(-1.1, 1.1)
    plt.tight_layout()
    plt.savefig("dtw.png")
    plt.clf()
    plt.close()
    return True



def compare_y(X, scale, gamma=1):
    seq_length = X.shape[1]
    num_signals = X.shape[2]
    Y = X + np.random.normal(size=(seq_length, num_signals), scale=scale)

    x = X[0, :, :]
    y = Y[0, :, :]

    kxy = my_rbf(x, y, gamma=gamma)
    print(kxy)

    plt.plot(x[:, 0], color='blue')
    plt.plot(x[:, 1], color='green')
    plt.plot(x[:, 2], color='red')
    plt.plot(y[:, 0], color='#4286f4')
    plt.plot(y[:, 1], color='#20cc4b')
    plt.plot(y[:, 2], color='#ea4b4b')
    plt.axhline(y=kxy, color='black', linestyle='-', label='kxy')
    plt.fill_between(plt.xlim(), 0, 1, facecolor='black', alpha=0.15)
    plt.title('gamma' + str(gamma) + ' scale' + str(scale).zfill(3))
    plt.xlim(0, seq_length-1)
    plt.ylim(-1.01, 1.01)
    #plt.ylim(4, 4)
    plt.savefig('sine_gamma' + str(gamma) + '_scale' + str(scale*100).zfill(5) + '.png')
    plt.clf()
    plt.close()

#for scale in np.concatenate(([5, 1, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1], np.arange(0.09, 0.00, -0.01))):
#    compare_y(X, scale, 0.1)
#    compare_y(X, scale, 0.5)
#    compare_y(X, scale, 1)
#    compare_y(X, scale, 2)
