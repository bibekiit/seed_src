from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import gaussian_function as gf
import pdb, math
#------------------------------------------------------------------------
#smoothen_frequency_image
#smoothens the frequency image through a process of diffusion
#Usage:
#new_oimg = smoothen_frequency_image(fimg,RLOW,RHIGH,diff_cycles)
#fimg       - frequency image image
#nimg       - filtered frequency image
#RLOW       - lowest allowed ridge separation
#RHIGH      - highest allowed ridge separation
#diff_cyles - number of diffusion cycles
#Contact:
#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#Reference:
#S. Chikkerur, C.Wu and V. Govindaraju, "Systematic approach for feature
#extraction in Fingerprint Images", ICBA 2004
#------------------------------------------------------------------------
def smoothen_frequency_image(fimg, RLOW, RHIGH, diff_cycles):
    valid_nbrs = 3
    #uses only pixels with more then valid_nbrs for diffusion
    ht, wt = fimg.shape # nargout=2
    nfimg = fimg.copy()
    N = 1
    #---------------------------------
    #perform diffusion
    #---------------------------------
    h = gf.matlab_style_gauss2D((np.dot(2, N) + 1,np.dot(2, N) + 1),0.5)
    cycles = 0
    invalid_cnt = np.sum(np.sum(np.logical_or(fimg < RLOW,fimg > RHIGH)))
    while (np.logical_or(np.logical_and(invalid_cnt > 0, cycles < diff_cycles), cycles < diff_cycles)):
        #---------------
        #pad the image
        #---------------
        fimg = np.vstack([fimg[0:N, :][ ::-1,:], fimg, fimg[(ht - N + 1 -1):ht, :][ ::-1,:]])
        #pad the rows
        fimg = np.hstack([fimg[:, 0:N][:, ::-1], fimg, fimg[:, (wt - N + 1 -1):wt][:, ::-1]])
        #pad the cols
        #---------------
        #perform diffusion
        #---------------
        for i in range(N + 1, (ht + N +1)):
            for j in range(N + 1, (wt + N +1)):
                blk = fimg[(i - N -1):i + N, (j - N -1):j + N]
                msk = np.logical_and(blk >= RLOW , blk <= RHIGH)
                if (np.sum(np.sum(msk)) >= valid_nbrs):
                    blk = blk * msk
                    nfimg[(i - N -1), (j - N -1)] = np.sum(np.sum(blk * h)) / np.sum(np.sum(h * msk))
                else:
                    nfimg[(i - N -1), (j - N -1)] = - 1
                    #invalid value
        #---------------
        #prepare for next iteration
        #---------------
        fimg = nfimg.copy()
        invalid_cnt = np.sum(np.sum(np.logical_or(fimg < RLOW,fimg > RHIGH)))
        cycles = cycles + 1
    print('cycles = '+ str(cycles))
    #end function smoothen_orientation_image
    return nfimg
##fimg = loadmat('fimg.mat')
##fimg = fimg['fimg']
##fimg    =   smoothen_frequency_image(fimg,4,20,5)
