#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
# This code is a python version of Peter Kovesi Matlab's code               #
# for finger print image enhancement                                        #
#############################################################################

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy
import scipy.ndimage as ssig
import pylab as pl
import numpy as np
from scipy.io import loadmat,savemat
import os
from PIL import Image


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# The function takes a grayscale image and the number of
# bins to use in the histogram as input and returns an
# image with equalized histogram together with the cumulative
# distribution function used to do the mapping of pixel values. 
def histeq(im,nbr_bins=256):

   #get image histogram
   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize

   #use linear interpolation of cdf to find new pixel values
   im2 = interp(im.flatten(),bins[:-1],cdf)

   return im2.reshape(im.shape), cdf

def segmented_process(M, blk_size=(16,16), overlap=(0,0), fun=None):
    rows = []
    for i in range(0, M.shape[0], blk_size[0]):
        cols = []
        for j in range(0, M.shape[1], blk_size[1]):
            cols.append(fun(M[i:i+blk_size[0], j:j+blk_size[1]]))
        rows.append(np.concatenate(cols, axis=1))
    return np.concatenate(rows, axis=0)


def normalise (im):
    im = (im - np.mean(im))/np.std(im) 
    return im


# RIDGESEGMENT - Normalises fingerprint image and segments ridge region
#
# Function identifies ridge regions of a fingerprint image and returns a
# mask identifying this region.  It also normalises the intesity values of
# the image so that the ridge regions have zero mean, unit standard
# deviation.
#
# This function breaks the image up into blocks of size blksze x blksze and
# evaluates the standard deviation in each region.  If the standard
# deviation is above the threshold it is deemed part of the fingerprint.
# Note that the image is normalised to have zero mean, unit standard
# deviation prior to performing this process so that the threshold you
# specify is relative to a unit standard deviation.
#
# Usage:   [normim, mask, maskind] = ridgesegment(im, blksze, thresh)
#
# Arguments:   im     - Fingerprint image to be segmented.
#              blksze - Block size over which the the standard
#                       deviation is determined (try a value of 16).
#              thresh - Threshold of standard deviation to decide if a
#                       block is a ridge region (Try a value 0.1 - 0.2)
#
# Returns:     normim - Image where the ridge regions are renormalised to
#                       have zero mean, unit standard deviation.
#              mask   - Mask indicating ridge-like regions of the image, 
#                       0 for non ridge regions, 1 for ridge regions.
#              maskind - Vector of indices of locations within the mask. 
#
# Suggested values for a 500dpi fingerprint image:
#
#   [normim, mask, maskind] = ridgesegment(im, 16, 0.1)
def ridgesegment(im, blksze, thresh):
    R = normalise(im);  # normalise to have zero mean, unit std dev
    passthrough = lambda(x):np.std(x)**np.ones(x.shape);
    stddevim = segmented_process(R, blk_size=(blksze,blksze), 
                           overlap=(0,0), 
                           fun=passthrough);
    mask = stddevim > thresh;
    maskind = np.nonzero(mask);
    
    
# Renormalise image so that the *ridge regions* have zero mean, unit
# standard deviation.
    
    im = im - np.mean(im.ravel()[maskind[0]])
    normim = im/np.std(im.ravel()[maskind[0]])
    
    return [normim, mask, maskind]



def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, b):
    [mx,nx] = x.shape
    stencil = np.rot90(b,2)
    [ms,ns] = stencil.shape
    y = ssig.convolve(x,stencil)
    return y
    

#RIDGEORIENT - Estimates the local orientation of ridges in a fingerprint
#
#Usage:  [orientim, reliability, coherence] = ridgeorientation(im, gradientsigma,...
#                                            blocksigma, ...
#                                            orientsmoothsigma)
#
#Arguments:  im                - A normalised input image.
#            gradientsigma     - Sigma of the derivative of Gaussian
#                                used to compute image gradients.
#            blocksigma        - Sigma of the Gaussian weighting used to
#                                sum the gradient moments.
#            orientsmoothsigma - Sigma of the Gaussian used to smooth
#                                the final orientation vector field. 
#                                Optional: if ommitted it defaults to 0
#
#Returns:    orientim          - The orientation image in radians.
#                                Orientation values are +ve clockwise
#                                and give the direction *along* the
#                                ridges.
#            reliability       - Measure of the reliability of the
#                                orientation measure.  This is a value
#                                between 0 and 1. I think a value above
#                                about 0.5 can be considered 'reliable'.
#                                reliability = 1 - Imin./(Imax+.001);
#            coherence         - A measure of the degree to which the local
#                                area is oriented.
#                                coherence = ((Imax-Imin)./(Imax+Imin)).^2;
#
#With a fingerprint image at a 'standard' resolution of 500dpi suggested
#parameter values might be:
#
#   [orientim, reliability] = ridgeorient(im, 1, 3, 3);

def ridgeorient(im, gradientsigma, blocksigma, orientsmoothsigma=0):
        
    [rows,cols] = im.shape;
    # Calculate image gradients.
    sze = int(6*gradientsigma);
    if (not (sze%2)):
        sze = sze+1;
    f = matlab_style_gauss2D(shape=(sze,sze),sigma=gradientsigma);   # Generate Gaussian filter.
    [fx,fy] = np.gradient(f);                         # Gradient of Gausian.
    Gx = filter2(im, fx.transpose());  # Gradient of the image in x
    Gy = filter2(im, fy.transpose());  # ... and y
    # Estimate the local ridge orientation at each point by finding the
    # principal axis of variation in the image gradients.
   
    Gxx = Gx**2;      # Covariance data for the image gradients
    Gxy = Gx*Gy;
    Gyy = Gy**2;

    # Now smooth the covariance data to perform a weighted summation of the
    # data.
    sze = int(6*blocksigma);
    if (not sze%2):
        sze = sze+1;  
    f = matlab_style_gauss2D(shape=(sze,sze),sigma=blocksigma);
    Gxx = filter2(Gxx, f);
    Gxy = 2*filter2(Gxy, f);
    Gyy = filter2(Gyy, f);

    # Analytic solution of principal direction
    eps = np.spacing(1);
    denom = np.sqrt(Gxy**2 + (Gxx - Gyy)**2) + eps;
    sin2theta = np.divide(Gxy,denom);            # Sine and cosine of doubled angles
    cos2theta = np.divide((Gxx-Gyy),denom);
    if orientsmoothsigma:
        sze = int(6*orientsmoothsigma)
        if (not sze%2):
            sze = sze+1
        f = matlab_style_gauss2D(shape=(sze,sze),sigma=orientsmoothsigma)
        cos2theta = filter2(cos2theta, f) # Smoothed sine and cosine of
        sin2theta = filter2(sin2theta, f) # doubled angles
    
    orientim = np.pi/2+(np.arctan2(sin2theta,cos2theta))/2
    
    # Calculate 'reliability' of orientation data.  Here we calculate the
    # area moment of inertia about the orientation axis found (this will
    # be the minimum inertia) and an axis  perpendicular (which will be
    # the maximum inertia).  The reliability measure is given by
    # 1.0-min_inertia/max_inertia.  The reasoning being that if the ratio
    # of the minimum to maximum inertia is close to one we have little
    # orientation information. 
    
    Imin = (Gyy+Gxx)/2 - np.multiply((Gxx-Gyy),cos2theta)/2 - np.multiply(Gxy,sin2theta)/2
    Imax = Gyy+Gxx - Imin

    reliability = 1 - np.divide(Imin,(Imax+.001))
    coherence = (np.divide(Imax-Imin,Imax+Imin))**2
    
    # Finally mask reliability to exclude regions where the denominator
    # in the orientation calculation above was small.  Here I have set
    # the value to 0.001, adjust this if you feel the need
    reliability = np.multiply ( reliability , denom > .001 )
    return [orientim, reliability]
                             
# PLOTRIDGEORIENT - plot of ridge orientation data
#
# Usage:   plotridgeorient(orient, spacing, im, figno)
#
#        orientim - Ridge orientation image (obtained from RIDGEORIENT)
#        spacing  - Sub-sampling interval to be used in ploting the
#                   orientation data the (Plotting every point is
#                   typically not feasible) 
#        im       - Optional fingerprint image in which to overlay the
#                   orientation plot.
#        figno    - Optional figure number for plot
#
# A spacing of about 20 is recommended for a 500dpi fingerprint image
#

def plotridgeorient(orient, spacing, im, figno=2):

    if (int(spacing) != spacing):
        print('spacing must be an integer')
      
    [rows, cols] = orient.shape
    
    lw = 2          # linewidth
    len = 0.8*spacing  # length of orientation lines

    # Subsample the orientation data according to the specified spacing

    s_orient = orient[spacing-1:rows-spacing:spacing,
		      spacing-1:cols-spacing:spacing]
    xoff = len/2*np.cos(s_orient)
    yoff = len/2*np.sin(s_orient)   
    plt.imshow(im, cmap = plt.get_cmap("gray"))
    
	
    # Determine placement of orientation vectors
    [x,y] = np.meshgrid(range(spacing,cols-spacing+1,spacing),
		      range(spacing,rows-spacing+1,spacing) )
    
    x = x-xoff
    y = y-yoff
    
    # Orientation vectors
    u = xoff*2
    v = yoff*2
    plt.quiver(x,y,v,u,linewidth=0.5)
    plt.show()

## FREQEST - Estimate fingerprint ridge frequency within image block
##
## Function to estimate the fingerprint ridge frequency within a small block
## of a fingerprint image.  This function is used by RIDGEFREQ
##
## Usage:
##  freqim =  freqest(im, orientim, windsze, minWaveLength, maxWaveLength)
##
## Arguments:
##         im       - Image block to be processed.
##         orientim - Ridge orientation image of image block.
##         windsze  - Window length used to identify peaks. This should be
##                    an odd integer, say 3 or 5.
##         minWaveLength,  maxWaveLength - Minimum and maximum ridge
##                     wavelengths, in pixels, considered acceptable.
## 
## Returns:
##         freqim    - An image block the same size as im with all values
##                     set to the estimated ridge spatial frequency.  If a
##                     ridge frequency cannot be found, or cannot be found
##                     within the limits set by min and max Wavlength
##                     freqim is set to zeros.
##
## Suggested parameters for a 500dpi fingerprint image
##   freqim = freqest(im,orientim, 5, 5, 15);


    
def freqest(im, orientim, windsze, minWaveLength, maxWaveLength):
    
    debug = 0
    
    [rows,cols] = im.shape
    
##     Find mean orientation within the block. This is done by averaging the
##     sines and cosines of the doubled angles before reconstructing the
##     angle again.  This avoids wraparound problems at the origin.
    orientim = 2*orientim.flatten(1)
    cosorient = np.mean(np.cos(orientim))
    sinorient = np.mean(np.sin(orientim))   
    orient = np.arctan2(sinorient,cosorient)/2

    # Rotate the image block so that the ridges are vertical
    rotim = ssig.interpolation.rotate(im,orient/np.pi*180+90,mode='nearest') # crop

    # Now crop the image so that the rotated image does not contain any
    # invalid regions.  This prevents the projection down the columns
    # from being mucked up.
    cropsze = int(rows/np.sqrt(2))
    offset = int((rows-cropsze)/2)
    rotim = rotim[offset-1:offset+cropsze, offset-1:offset+cropsze]
    
    # Sum down the columns to get a projection of the grey values down
    # the ridges.
    proj = rotim.sum(axis=0)
    
    # Find peaks in projected grey values by performing a greyscale
    # dilation and then finding where the dilation equals the original
    # values. 
    s = ssig.generate_binary_structure(1,windsze)
    dilation = ssig.grey_dilation(proj,size=(1,windsze), footprint=s)
    maxpts = (dilation == proj) & (proj > np.mean(proj))
    maxind = np.nonzero(maxpts)

    # Determine the spatial frequency of the ridges by divinding the
    # distance between the 1st and last peaks by the (No of peaks-1). If no
    # peaks are detected, or the wavelength is outside the allowed bounds,
    # the frequency image is set to 0
    if (maxind[0].shape[0] < 2):
        freqim = np.zeros(im.shape)
    else:
        NoOfPeaks = maxind[0].shape[0]
        waveLength = (maxind[0][-1]-maxind[0][0])/(NoOfPeaks-1)
        if (waveLength > minWaveLength) and waveLength < maxWaveLength :
            freqim = 1/waveLength * np.ones(im.shape)
        else:
            freqim = np.zeros(im.shape)
##    if max(maxind.shape) < 2:
##        freqim = np.zeros(im.shape)
##    else:
##        NoOfPeaks = max(maxind.shape)
##        waveLength = (maxind[-1] - maxind[0]) / (NoOfPeaks - 1)
##        if waveLength > minWaveLength and waveLength < maxWaveLength:
##            freqim = np.dot(1 / waveLength, np.ones(im.shape))
##        else:
##            freqim = np.zeros(im.shape)
    if debug:
        #show(im,1)
        #show(rotim,2);
        figure(3)
        plot(proj)
        hold('on')
        meanproj = mean(proj)
        if max(maxind.shape) < 2:
            fprintf('No peaks found\\n')
        else:
            plot(maxind, dilation[(maxind -1)], 'r*')
            hold('off')
            waveLength = (maxind[-1] - maxind[0]) / (NoOfPeaks - 1)

    return freqim


## RIDGEFREQ - Calculates a ridge frequency image
##
## Function to estimate the fingerprint ridge frequency across a
## fingerprint image. This is done by considering blocks of the image and
## determining a ridgecount within each block by a call to FREQEST.
##
## Usage:
##  [freqim, medianfreq] =  ridgefreq(im, mask, orientim, blksze, windsze, ...
##                                    minWaveLength, maxWaveLength)
##
## Arguments:
##         im       - Image to be processed.
##         mask     - Mask defining ridge regions (obtained from RIDGESEGMENT)
##         orientim - Ridge orientation image (obtained from RIDGORIENT)
##         blksze   - Size of image block to use (say 32) 
##         windsze  - Window length used to identify peaks. This should be
##                    an odd integer, say 3 or 5.
##         minWaveLength,  maxWaveLength - Minimum and maximum ridge
##                     wavelengths, in pixels, considered acceptable.
## 
## Returns:
##         freqim     - An image  the same size as im with  values set to
##                      the estimated ridge spatial frequency within each
##                      image block.  If a  ridge frequency cannot be
##                      found within a block, or cannot be found within the
##                      limits set by min and max Wavlength freqim is set
##                      to zeros within that block.
##         medianfreq - Median frequency value evaluated over all the
##                      valid regions of the image.
##
## Suggested parameters for a 500dpi fingerprint image
##   [freqim, medianfreq] = ridgefreq(im,orientim, 32, 5, 5, 15);

    
def ridgefreq(im, mask, orient, blksze, windsze, minWaveLength, maxWaveLength):
    rows, cols = im.shape # nargout=2
    freq = np.zeros(im.shape)
    for r in range(1, (rows - blksze +1), blksze):
        for c in range(1, (cols - blksze +1), blksze):
            blkim = im[(r -1):r + blksze - 1, (c -1):c + blksze - 1]
            blkor = orient[(r -1):r + blksze - 1, (c -1):c + blksze - 1]
            freq[(r -1):r + blksze - 1, (c -1):c + blksze - 1] = freqest(blkim, blkor, windsze, minWaveLength, maxWaveLength)
            
    # Mask out frequencies calculated for non ridge regions
    freq = freq * mask
    x,y = np.nonzero(freq)
    # Find median freqency over all the valid regions of the image.
    medianfreq = np.median(freq[x,y])
    return freq, medianfreq
    

# RIDGEFILTER - enhances fingerprint image via oriented filters
#
# Function to enhance fingerprint image via oriented filters
#
# Usage:
#  newim =  ridgefilter(im, orientim, freqim, kx, ky, showfilter)
#
# Arguments:
#         im       - Image to be processed.
#         orientim - Ridge orientation image, obtained from RIDGEORIENT.
#         freqim   - Ridge frequency image, obtained from RIDGEFREQ.
#         kx, ky   - Scale factors specifying the filter sigma relative
#                    to the wavelength of the filter.  This is done so
#                    that the shapes of the filters are invariant to the
#                    scale.  kx controls the sigma in the x direction
#                    which is along the filter, and hence controls the
#                    bandwidth of the filter.  ky controls the sigma
#                    across the filter and hence controls the
#                    orientational selectivity of the filter. A value of
#                    0.5 for both kx and ky is a good starting point.
#         showfilter - An optional flag 0/1.  When set an image of the
#                      largest scale filter is displayed for inspection.
# 
# Returns:
#         newim    - The enhanced image
#

def ridgefilter(*varargin):
    nargin = len(varargin)
    if nargin > 0:
        im = varargin[0]
    if nargin > 1:
        orient = varargin[1]
    if nargin > 2:
        freq = varargin[2]
    if nargin > 3:
        kx = varargin[3]
    if nargin > 4:
        ky = varargin[4]
    if nargin > 5:
        showfilter = varargin[5]
    if nargin == 5:
        showfilter = 0
    angleInc = 3;
    # Fixed angle increment between filter orientations in
    # degrees. This should divide evenly into 180
    im.astype('double');
    rows, cols = im.shape; # nargout=2
    newim = np.zeros(shape=(rows, cols), dtype='float64');
    
    validr, validc = np.nonzero(freq); # nargout=2
    
    # find where there is valid frequency data.
    # Round the array of frequencies to the nearest 0.01 to reduce the
    # number of distinct frequencies we have to deal with.
    freq[validr, validc] = np.around(freq[validr, validc], 2);
    
    # Generate an array of the distinct frequencies present in the array
    # freq 
    unfreq = np.unique(freq[validr, validc]);
    
    # Generate a table, given the frequency value multiplied by 100 to obtain
    # an integer index, returns the index within the unfreq array that it
    # corresponds to
    freqindex = np.ones(shape=(100, 1), dtype='float64');
    
    for k in range(1, (max(unfreq.shape) +1)):
        freqindex[np.around(np.dot(unfreq[k -1], 100) -1)] = k;
    
    # Generate filters corresponding to these distinct frequencies and
    # orientations in 'angleInc' increments.
    filter1 = {};
    for i in range(1,max(unfreq.shape)+1):
        filter1[i-1] = {};
        for j in range(1,int(180 / angleInc) + 1):
            filter1[i-1][j-1] = {};
    
    sze = np.zeros(shape=(max(unfreq.shape), 1), dtype='float64');
    for k in range(1, (max(unfreq.shape) +1)):
        sigmax = np.dot(1 / unfreq[(k -1)], kx);
        sigmay = np.dot(1 / unfreq[(k -1)], ky);
        sze[(k -1)] = np.around(np.dot(3, max(sigmax, sigmay)));
        y,x = np.mgrid[(-sze[k-1]):(sze[k-1]+1),(-sze[k-1]):(sze[k-1]+1)]; # nargout=2
        reffilter = np.exp(- (x ** 2 / sigmax ** 2 + y ** 2 / sigmay ** 2) / 2) * np.cos(2*np.pi*unfreq[(k -1)]*x);
        
        # Generate rotated versions of the filter.  Note orientation
        # image provides orientation *along* the ridges, hence +90
        # degrees, and imrotate requires angles +ve anticlockwise, hence
        # the minus sign.
        for o in range(1, int(180 / angleInc +1)):
            filter1[(k -1)][(o -1)] = ssig.interpolation.rotate(reffilter, - ((o*angleInc) + 90), mode='nearest',reshape=False); # 'crop');
            
    if showfilter:
        # Display largest scale filter for inspection
        plt.figure(7);
        plt.imshow(filter1[0][180 / angleInc -1], cmap = plt.get_cmap("gray")); # cmap = plt.get_cmap("gray"));
        plt.title('filter');
        plt.show()
        
    # Find indices of matrix points greater than maxsze from the image
    # boundary
    maxsze = sze[0];
    finalind = np.nonzero((validr > maxsze) & (validr < rows - maxsze) & (validc > maxsze) & (validc < cols - maxsze));
    
    # Convert orientation matrix values from radians to an index value
    # that corresponds to round(degrees/angleInc)
    maxorientindex = round(180 / angleInc);
    orientindex = (np.dot(orient / np.pi, 180) / angleInc).astype(int);
    i, j = np.nonzero(orientindex < 1);
    orientindex[i,j] = orientindex[i,j] + maxorientindex;
    i, j = np.nonzero(orientindex > maxorientindex);
    orientindex[i,j] = orientindex[i,j] - maxorientindex;
    # Finally do the filtering
    
    for k in range(1, (max(finalind[0].shape) +1)):
        
        r = validr[(finalind[0][(k -1)])];
        c = validc[(finalind[0][(k -1)])];

        # find filter corresponding to freq(r,c)
        filterindex = freqindex[(freq[(r -1), (c -1)]*100).astype(int)];
        s = sze[(filterindex[0] -1).astype(int)];
        newim[(r -1), (c -1)] = np.sum(im[(r - s -1):(r + s), (c - s -1):(c + s)] * filter1[(filterindex[0] -1)][orientindex[(r -1), (c -1)]-1]);
    return newim

im = mpimg.imread("C:\Users\Bibek\Documents\MATLAB\sc_minutia\sc_minutia\\108_2.tif")     
#im = np.around(rgb2gray(img))
plt.imshow(im, cmap = plt.get_cmap("gray")) # this cmap is not clear
plt.show()

# Histogram equalization 
im2,cdf = histeq(im)
plt.imshow(im, cmap = plt.get_cmap("gray")) # this cmap is not clear
plt.show()

# Identify ridge-like regions and normalise image
[normim, mask, maskind] = ridgesegment(im, blksze=16, thresh=0.1)
plt.figure(1)
plt.imshow(normim, cmap = plt.get_cmap("gray"))
plt.title('normalized image')
plt.show()

# Determine ridge orientations
[orientim, reliability] = ridgeorient(normim, 1, 5, 5)
plotridgeorient(orientim, 20, im, 2)
plt.figure(6)
plt.imshow(reliability, cmap = plt.get_cmap("gray"))
plt.title('reliability')
plt.show()



# Determine ridge frequency values across the image
blksze = 36; 
[freq, medfreq] = ridgefreq(normim, mask, orientim, blksze, 5, 5, 15);
plt.figure(3)
plt.imshow(freq, cmap = plt.get_cmap("gray"))
plt.title('frequency image')
plt.show()

# Actually I find the median frequency value used across the whole
# fingerprint gives a more satisfactory result...
freq = medfreq*mask;


# Now apply filters to enhance the ridge pattern
newim = ridgefilter(normim, orientim, freq, 0.5, 0.5, 1);
plt.figure(4)
plt.imshow(newim, cmap = plt.get_cmap("gray"))
plt.title('image after applying gabor filter')
plt.show()

# Binarise, ridge/valley threshold is 0
binim = newim > 0;
plt.figure(5)
plt.imshow(binim, cmap = plt.get_cmap("gray"))
plt.title('binarised image')
plt.show()

# Display binary image for where the mask values are one and where
# the orientation reliability is greater than 0.5
plt.figure(7)
plt.imshow(binim*mask*(reliability>0.5), cmap = plt.get_cmap("gray"))
plt.show()
