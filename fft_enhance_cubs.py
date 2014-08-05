from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import pdb, math
import smoothen_orientation_image as soi
import smoothen_frequency_image as sfi
import compute_coherence as cc
import matplotlib.pyplot as plt
import scipy 

#--------------------------------------------------------------------------
#fft_enhance_cubs
#enhances the fingerprint image
#syntax:
#[oimg,fimg,bwimg,eimg,enhimg] =  fft_enhance_cubs(img, BLKSZ)
#oimg -  [OUT] block orientation image(can be viewed using
#        view_orientation_image.m)
#fimg  - [OUT] block frequency image(indicates ridge spacing)
#bwimg - [OUT] shows angular bandwidth image(filter bandwidth adapts near the
#        singular points)
#eimg  - [OUT] energy image. Indicates the 'ridgeness' of a block (can be 
#        used for fingerprint segmentation)
#enhimg- [OUT] enhanced image
#img   - [IN]  input fingerprint image (HAS to be of DOUBLE type)
#Contact:
#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#Reference:
#S. Chikkerur,C. Wu and Govindaraju, "A Systematic approach for 
#feature extraction in fingerprint images",ICBA 2004
#--------------------------------------------------------------------------
def fft_enhance_cubs(img, BLKSZ):
    global NFFT
    if BLKSZ > 0:
        NFFT = 32
        #size of FFT
        OVRLP = 2
        #size of overlap
        ALPHA = 0.5
        #root filtering
        RMIN = 4
        #3;      
        #min allowable ridge spacing
        RMAX = 40
        #maximum allowable ridge spacing
        ESTRETCH = 20
        #for contrast enhancement
        ETHRESH = 19
        #threshold for the energy
    else:
        NFFT = 32
        #size of FFT
        BLKSZ = 12
        #size of the block
        OVRLP = 6
        #size of overlap
        ALPHA = 0.5
        #root filtering
        RMIN = 3
        #min allowable ridge spacing
        RMAX = 18
        #maximum allowable ridge spacing
        ESTRETCH = 20
        #for contrast enhancement
        ETHRESH = 6
        #threshold for the energy

    nHt, nWt = img.shape # nargout=2
    img = img.astype('float64') #convert to DOUBLE
    nBlkHt = math.floor((nHt - np.dot(2, OVRLP)) / BLKSZ) 
    nBlkWt = math.floor((nWt - np.dot(2, OVRLP)) / BLKSZ) 
    fftSrc = np.zeros(shape=(np.dot(nBlkHt, nBlkWt), np.dot(NFFT, NFFT)), dtype='float64')#stores FFT
    nWndSz = BLKSZ + np.dot(2, OVRLP) #size of analysis window. 
    #-------------------------
    #allocate outputs
    #-------------------------
    oimg = np.zeros(shape=(nBlkHt, nBlkWt), dtype='float64')
    fimg = np.zeros(shape=(nBlkHt, nBlkWt), dtype='float64')
    bwimg = np.zeros(shape=(nBlkHt, nBlkWt), dtype='float64')
    eimg = np.zeros(shape=(nBlkHt, nBlkWt), dtype='float64')
    enhimg = np.zeros(shape=(nHt, nWt), dtype='float64')

    #-------------------------
    #precomputations
    #-------------------------
    x, y = np.meshgrid(range(0, (nWndSz - 1 +1)), range(0, (nWndSz - 1 +1))) # nargout=2
    dMult = (- 1) ** (x + y)#used to center the FFT
    x, y = np.meshgrid(np.arange(- NFFT / 2, (NFFT / 2 - 1 +1)), np.arange(- NFFT / 2, (NFFT / 2 - 1 +1))) # nargout=2
    eps = np.spacing(1)
    r = np.sqrt(x ** 2 + y ** 2) + eps
    th = np.arctan2(y, x)
    th[th < 0] = th[th < 0] + np.pi
    w = raised_cosine_window(BLKSZ, OVRLP)#spectral window

    #-------------------------
    #Load filters
    #-------------------------
    angf = loadmat('angular_filters_pi_4',matlab_compatible=True)['angf']#now angf_pi_4 has filter coefficients
    angf_pi_4 = angf
    angf = loadmat('angular_filters_pi_2',matlab_compatible=True)['angf']#now angf_pi_2 has filter coefficients
    angf_pi_2 = angf
    #-------------------------
    #Bandpass filter
    #-------------------------
    FLOW = NFFT / float(RMAX)
    FHIGH = NFFT / float(RMIN)
    dRLow = 1.0 / (1 + (r / FHIGH) ** 4)#low pass butterworth filter
    dRHigh = 1.0 / (1 + (FLOW / r) ** 4)#high pass butterworth filter
    dBPass = dRLow * dRHigh#bandpass

    #-------------------------
    #FFT Analysis
    #-------------------------

    fftSrc = fftSrc + 0j# makes fftSrc complex for later assignment from complex matrix blkfft
    for i in range(0, (int(nBlkHt) - 1 +1)):
        nRow = np.dot(i, BLKSZ) + OVRLP + 1
        for j in range(0, (int(nBlkWt) - 1 +1)):
            nCol = np.dot(j, BLKSZ) + OVRLP + 1
            #extract local block
            blk = img[(nRow - OVRLP -1):nRow + BLKSZ + OVRLP - 1][:, (nCol - OVRLP -1):nCol + BLKSZ + OVRLP - 1]#remove dc
            dAvg = np.sum(blk) / (np.dot(nWndSz, nWndSz))
            blk = blk - dAvg#remove DC content
            blk = blk * w#multiply by spectral window
            #--------------------------
            #do pre filtering
            #--------------------------
            blkfft = np.fft.fft2(blk * dMult, (NFFT, NFFT))
            blkfft = blkfft * dBPass#band pass filtering
            dEnergy = abs(blkfft * blkfft)
            blkfft = blkfft * np.sqrt(dEnergy)#root filtering(for diffusion)
            fftSrc[(np.dot(nBlkWt, i) + j + 1 -1), :] = blkfft.copy().flatten(1)
            dEnergy = abs(blkfft * blkfft)#----REDUCE THIS COMPUTATION----
            #--------------------------
            #compute statistics
            #--------------------------

            dTotal = np.sum(np.sum(dEnergy)) / (np.dot(NFFT, NFFT))
            
            fimg[(i + 1 -1), (j + 1 -1)] = NFFT / (compute_mean_frequency(dEnergy, r) + eps)#ridge separation
            oimg[(i + 1 -1), (j + 1 -1)] = compute_mean_angle(dEnergy, th)#ridge angle
            eimg[(i + 1 -1), (j + 1 -1)] = np.log(dTotal + eps)#np.spacing(1) is a small number taken to avoid log(0)#used for segmentation

    #-------------------------
    #precomputations
    #-------------------------
    x, y = np.meshgrid(np.arange(- NFFT / 2, (NFFT / 2 - 1 +1)), np.arange(- NFFT / 2, (NFFT / 2 - 1 +1))) # nargout=2
    dMult = (- 1) ** (x + y)#used to center the FFT

    #-------------------------
    #process the resulting maps
    #-------------------------
    for i in range(1, 4):
        oimg = soi.smoothen_orientation_image(oimg)#smoothen orientation image
    fimg = sfi.smoothen_frequency_image(fimg, RMIN, RMAX, 5)#diffuse frequency image
    cimg = cc.compute_coherence(oimg)#coherence image for bandwidth
    bwimg = get_angular_bw_image(cimg)#QUANTIZED bandwidth image
    #-------------------------
    #FFT reconstruction
    #-------------------------

##    oimg = loadmat('oimg.mat')['oimg']
##    fftSrc = loadmat('fftSrc.mat')['fftSrc']
    
    for i in range(0, (int(nBlkHt) - 1 +1)):
        for j in range(0, (int(nBlkWt) - 1 +1)):
            nRow = np.dot(i, BLKSZ) + OVRLP + 1
            nCol = np.dot(j, BLKSZ) + OVRLP + 1
            #--------------------------
            #apply the filters
            #--------------------------
            blkfft = np.reshape(np.tile(fftSrc[(np.dot(nBlkWt, i) + j + 1 -1), :],(1,1)).T,(NFFT,NFFT)).T
            #--------------------------
            #reconstruction
            #--------------------------
            af = get_angular_filter(oimg[(i + 1 -1), (j + 1 -1)], bwimg[(i + 1 -1), (j + 1 -1)], angf_pi_4, angf_pi_2)
            blkfft = blkfft * (af)
            blk = np.real(np.fft.ifft2(blkfft) * dMult)
            enhimg[np.ix_(range((nRow -1),nRow + BLKSZ - 1),range((nCol -1),nCol + BLKSZ - 1))] = blk[(OVRLP + 1 -1):OVRLP + BLKSZ][:,(OVRLP + 1 -1):OVRLP + BLKSZ].copy()
    #end block processing
    #--------------------------
    #contrast enhancement
    #--------------------------
    enhimg = np.sqrt(abs(enhimg)) * np.sign(enhimg)
    mx = np.max(enhimg)
    mn = np.min(enhimg)
    enhimg = (np.dot((enhimg - mn) / (mx - mn), 254) + 1).astype('uint8')
    #--------------------------
    #clean up the image
    #--------------------------
    emsk = scipy.misc.imresize(eimg, (nHt, nWt))
    enhimg[emsk < ETHRESH] = 128

    return [cimg, oimg, fimg, bwimg, eimg, enhimg]
    #end function fft_enhance_cubs

#-----------------------------------
#raised_cosine
#returns 1D spectral window
#syntax:
#y = raised_cosine(nBlkSz,nOvrlp)
#y      - [OUT] 1D raised cosine function
#nBlkSz - [IN]  the window is constant here
#nOvrlp - [IN]  the window has transition here
#-----------------------------------
    
def raised_cosine(nBlkSz, nOvrlp):
    nWndSz = (nBlkSz + np.dot(2, nOvrlp))
    x = abs(np.arange(- nWndSz / 2, (nWndSz / 2 - 1 +1)))
    y = np.dot(0.5, (np.cos(np.dot(np.pi, (x - nBlkSz / 2)) / nOvrlp) + 1))
    y[abs(x) < nBlkSz / 2] = 1
    return y
    #end function raised_cosine
    #-----------------------------------
    #raised_cosine_window
    #returns 2D spectral window
    #syntax:
    #w = raised_cosine_window(blksz,ovrlp)
    #w      - [OUT] 1D raised cosine function
    #nBlkSz - [IN]  the window is constant here
    #nOvrlp - [IN]  the window has transition here
    #-----------------------------------
    
def raised_cosine_window(blksz, ovrlp):
    y = raised_cosine(blksz, ovrlp)
    w = np.dot(np.tile(y,(1,1)).T, np.tile(y,(1,1)))
    return w
    #end function raised_cosine_window
    #---------------------------------------------------------------------
    #get_angular_filter
    #generates an angular filter centered around 'th' and with bandwidth 'bw'
    #the filters in angf_xx are precomputed using angular_filter_bank.m
    #syntax:
    #r = get_angular_filter(t0,bw)
    #r - [OUT] angular filter of size NFFTxNFFT
    #t0- mean angle (obtained from orientation image)
    #bw- angular bandwidth(obtained from bandwidth image)
    #angf_xx - precomputed filters (using angular_filter_bank.m)
    #-----------------------------------------------------------------------
    
def get_angular_filter(t0, bw, angf_pi_4, angf_pi_2):
    global NFFT
    TSTEPS = angf_pi_4.shape[1]
    DELTAT = np.pi / TSTEPS
    #get the closest filter
    i = np.floor((t0 + DELTAT / 2) / DELTAT)
    i = np.mod(i, TSTEPS) + 1
    if (bw == np.pi / 4):
        r = np.reshape(angf_pi_4[:, (i -1)], (NFFT, NFFT))
    else:
        if (bw == np.pi / 2):
            r = np.reshape(angf_pi_2[:, (i -1)], (NFFT, NFFT))
        else:
            r = np.ones(shape=(NFFT, NFFT), dtype='float64')
    return r
    #end function get_angular_filter
    #-----------------------------------------------------------
    #get_angular_bw_image
    #the bandwidth allocation is currently based on heuristics
    #(domain knowledge :)). 
    #syntax:
    #bwimg = get_angular_bw_image(c)
    #-----------------------------------------------------------
    
def get_angular_bw_image(c):
    bwimg = np.zeros(shape=c.shape, dtype='float64')
    bwimg[:, :] = np.pi / 2#med bw
    bwimg[(c <= 0.7 -1)] = np.pi#high bw
    bwimg[c >= 0.9] = np.pi / 4#low bw
    return bwimg
    #end function get_angular_bw
    #-----------------------------------------------------------
    #get_angular_bw_image
    #the bandwidth allocation is currently based on heuristics
    #(domain knowledge :)). 
    #syntax:
    #bwimg = get_angular_bw_image(c)
    #-----------------------------------------------------------
    
def compute_mean_angle(dEnergy, th):
    global NFFT
    sth = np.sin(np.dot(2, th))
    cth = np.cos(np.dot(2, th))
    num = np.sum(dEnergy * sth)
    den = np.sum(dEnergy * cth)
    mth = np.dot(0.5, math.atan2(num, den))
    if (mth < 0):
        mth = mth + np.pi
    return mth
    #end function compute_mean_angle
    #-----------------------------------------------------------
    #get_angular_bw_image
    #the bandwidth allocation is currently based on heuristics
    #(domain knowledge :)). 
    #syntax:
    #bwimg = get_angular_bw_image(c)
    #-----------------------------------------------------------
    
def compute_mean_frequency(dEnergy, r):
    global NFFT
    num = np.sum(dEnergy * r)
    den = np.sum(dEnergy)
    mr = num / (den + np.spacing(1))
    return mr
    #end function compute_mean_angle
    

# Testing section #################
                                 
##def main():
##img = loadmat('enhimg.mat')        #
##img = img['img']                #
##fft_enhance_cubs(img, 6)        #
##
###########TESTING############
##thread = threading.Thread()
##thread.run = main
##
##manager = plt.get_current_fig_manager()
##manager.window.after(100, thread.start)
##plt.figure(1)
##plt.show()

