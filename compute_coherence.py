from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import pdb
#------------------------------------------------------------------------
#compute_coherence
#Computes the coherence image. 
#Usage:
#[cimg] = compute_coherence(oimg)
#oimg - orientation image
#cimg - coherence image(0-low coherence,1-high coherence)
#Contact:
#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#Reference:
#A. Ravishankar Rao,"A taxonomy of texture description", Springer Verlag
#------------------------------------------------------------------------
def compute_coherence(oimg):
    h, w = oimg.shape # nargout=2
    cimg = np.zeros(shape=(h, w), dtype='float64')
    N = 2
    #---------------
    #pad the image
    #---------------
    oimg = np.vstack([oimg[0:N, :][ ::-1,:], oimg, oimg[(h - N + 1 -1):h, :][ ::-1,:]])     #pad the rows
    oimg = np.hstack([oimg[:, 0:N][:, ::-1], oimg, oimg[:, (w - N + 1 -1):w][:, ::-1]])    #pad the cols
    #compute coherence
    for i in range(N + 1, (h + N +1)):
        for j in range(N + 1, (w + N +1)):
            th = oimg[(i -1), (j -1)]
            blk = oimg[(i - N -1):i + N, (j - N -1):j + N].copy()
            cimg[(i - N -1), (j - N -1)] = np.sum(np.sum(abs(np.cos(blk - th)))) / ((np.dot(2, N) + 1) ** 2)
    #end function compute_coherence
    return cimg

##oimg = loadmat('oimg.mat')
##oimg = oimg['oimg']
##cimg = compute_coherence(oimg)
