#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
from __future__ import division
import numpy as np
import scipy.io
from scipy.io import loadmat,savemat
import os
import math
import dist2 as ds

def bookstein(*varargin):
    """[cx,cy,E,L]=bookstein(X,Y,beta_k);

     Bookstein PAMI89
    """

    nargin = len(varargin)
    if nargin > 0:
        X = varargin[0]
    if nargin > 1:
        Y = varargin[1]
    if nargin > 2:
        beta_k = varargin[2]

    N = X.shape[0]
    Nb = Y.shape[0]

    if N != Nb:
        error('number of landmarks must be equal')

    # compute euclidean distances ^2 between left points
    r2 = ds.dist2(X, X)
    r2 = r2 + np.spacing(1)
    # add identity matrix to make K zero on the diagonal
    #JI since log(1) = 0

    K = r2 * np.log(r2 + np.eye(N))


    #JI: ith row of P = (1, xi, yi) 
    P = np.concatenate((np.ones(shape=(N, 1), dtype='float16'), X),1)

    L = np.concatenate((np.concatenate((K, P),1), np.concatenate((P.T, np.zeros(shape=(3, 3), dtype='float16')),1)))
    V = np.concatenate((Y.T, np.zeros(shape=(2, 3), dtype='float16')),1)

    if nargin > 2:
        # regularization
        L[0:N, 0:N] = L[0:N, 0:N] + np.dot(beta_k, np.eye(N))

    if np.flatnonzero(np.isnan(L)).size > 0:
        cx = 0
        cy = 0
        E = 100
        L = 0
        return cx, cy, E, L

    invL = np.linalg.inv(L)
    c = np.dot(invL, V.T)

    cx = c[:, 0]
    cy = c[:, 1]

    # compute bending energy (w/o regularization)
    Q = np.dot(np.dot(c[0:N, :].T, K), c[0:N, :])
    E = np.mean(np.sum(np.diag(Q)) + np.dot(2, abs(Q[0, 1])))
    print 'E = ' + str(E) + '\n'
    #   E=mean(diag(Q));
    return [cx, cy, E]

##X = scipy.io.loadmat("C:\Users\Bibek\Documents\MATLAB\DB1_B\X3b.mat")
##Y = scipy.io.loadmat("C:\Users\Bibek\Documents\MATLAB\DB1_B\Y3.mat")
##beta_k = 0.0072
##[cx,cy,E] =bookstein(X[X.keys()[0]],Y[Y.keys()[0]],beta_k);

