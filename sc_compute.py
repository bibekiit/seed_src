from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import scipy.io
from scipy.sparse import bsr_matrix
import math
import dist2 as ds

#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################

def sc_compute(Bsamp, Tsamp, mean_dist, nbins_theta, nbins_r, r_inner, r_outer, out_vec):
    """[BH,mean_dist]=sc_compute(Bsamp,Tsamp,mean_dist,nbins_theta,nbins_r,r_inner,r_outer,out_vec);

     compute (r,theta) histograms for points along boundary 

     Bsamp is 2 x nsamp (x and y coords.)
     Tsamp is 1 x nsamp (tangent theta)
     out_vec is 1 x nsamp (0 for inlier, 1 for outlier)

     mean_dist is the mean distance, used for length normalization
     if it is not supplied, then it is computed from the data

     outliers are not counted in the histograms, but they do get
     assigned a histogram

    """
    nsamp = Bsamp.shape[1]
    in_vec = out_vec == 0
    # compute r,theta arrays
    r_array = np.real(np.sqrt(ds.dist2(Bsamp.T, Bsamp.T)))
    # real is needed to
    # prevent bug in Unix version
    #JI: This is an array with elements arctan(xi-xj, yi-yj)
    theta_array_abs = np.arctan2(np.tile(Bsamp[1, :].T,(1,1)).T * np.ones(shape=(1, nsamp), dtype='float16') - np.ones(shape=(nsamp, 1), dtype='float16')* np.tile(Bsamp[1, :],(1,1)), np.tile(Bsamp[0, :],(1,1)).T * np.ones(shape=(1, nsamp), dtype='float16') - np.ones(shape=(nsamp, 1), dtype='float16')* np.tile(Bsamp[0, :],(1,1))).T
    theta_array = theta_array_abs - np.tile(Tsamp,(1,1)).T * np.ones(shape=(1, nsamp), dtype='float16')
    # create joint (r,theta) histogram by binning r_array and
    # theta_array
    # normalize distance by mean, ignoring outliers
    if (np.size(mean_dist) == 0):
        tmp = r_array[in_vec.nonzero()[1],:]
        #JI: removes rows not in in_vec
        tmp = tmp[:, in_vec.nonzero()[1]]
        #JI: removes cols not in in_vec
        mean_dist = np.mean(tmp[:])
        #JI: global mean
    r_array_n = r_array / mean_dist
    # use a log. scale for binning the distances
    r_bin_edges = np.logspace(np.math.log10(r_inner), np.math.log10(r_outer), nbins_r)
    r_array_q = np.zeros(shape=(nsamp, nsamp), dtype='float16')
    for m in range(1, (nbins_r +1)):
        r_array_q = r_array_q + (r_array_n < r_bin_edges[(m -1)])
    fz = r_array_q > 0
    # flag all points inside outer boundary
    # put all angles in [0,2pi) range
    theta_array_2 = np.remainder(np.remainder(theta_array, np.dot(2, np.pi)) + np.dot(2, np.pi), np.dot(2, np.pi))
    # quantize to a fixed set of angles (bin edges lie on 0,(2*pi)/k,...2*pi
    theta_array_q = 1 + np.floor(theta_array_2 / (np.dot(2, np.pi) / nbins_theta))
    #JI:
    #Gaussian array
    gauss_dist = np.tile(np.tile(np.exp(- ((r_bin_edges - r_outer) ** 2 / 2.5)),(1,1)).T, (1, nbins_theta))
    gauss_dist = np.sort(gauss_dist.flatten(1))
    nbins = np.dot(nbins_theta, nbins_r)
    BH = np.zeros(shape=(nsamp, nbins), dtype='float16')
    for n in range(1, (nsamp +1)):
        fzn = np.logical_and(fz[(n -1), :] , in_vec)
        Sn1 = bsr_matrix((np.ones(shape=(1, np.sum(np.logical_and(fz[(n -1), :] , in_vec)==True)), dtype='float16')[0,:],(theta_array_q[(n -1),np.nonzero(fzn)[1]]-1,r_array_q[(n -1),np.nonzero(fzn)[1]]-1))).todense()
        zeroR = np.zeros((nbins_theta-np.shape(Sn1)[0],np.shape(Sn1)[1]))
        Sn1 = np.concatenate((Sn1,zeroR))
        zeroC=np.zeros((nbins_theta,nbins_r-np.shape(Sn1)[1]))
        Sn = np.concatenate((Sn1,zeroC),1)
        BH[(n -1), :] = np.multiply(Sn.flatten(1)[0,:] , np.tile(gauss_dist,(1,1))[0,:])
    return BH, mean_dist
##Bsamp = scipy.io.loadmat("C:\Users\Bibek\Documents\MATLAB\DB1_B\Bsamp.mat")
##Tsamp = scipy.io.loadmat("C:\Users\Bibek\Documents\MATLAB\DB1_B\Tsamp.mat")
##out_vec = scipy.io.loadmat("C:\Users\Bibek\Documents\MATLAB\DB1_B\OUT_VEC.mat")
##[BH,mean_dist]=sc_compute(Bsamp[Bsamp.keys()[0]],Tsamp[Tsamp.keys()[2]],0.2784,19,5,0.125,0.5,out_vec[out_vec.keys()[1]]);
##print BH
