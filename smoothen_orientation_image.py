from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import filter2 as f2
import gaussian_function as gf

#------------------------------------------------------------------------
#smoothen_orientation_image
#smoothens the orientation image through vectorial gaussian filtering
#Usage:
#new_oimg = smoothen_orientation_image(oimg)
#oimg     - orientation image
#new_oimg - filtered orientation image
#Contact:
#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#Reference:
#M. Kaas and A. Witkin, "Analyzing oriented patterns", Computer Vision
#Graphics Image Processing 37(4), pp 362--385, 1987
#------------------------------------------------------------------------

def smoothen_orientation_image(oimg):
    """---------------------------
    smoothen the image
    ---------------------------
    """
    gx = np.cos(np.dot(2, oimg))
    gy = np.sin(np.dot(2, oimg))
    msk = gf.matlab_style_gauss2D((5,5),0.5)
    gfx = f2.filter2(gx, msk)
    gfy = f2.filter2(gy, msk)
    noimg = np.arctan2(gfy, gfx)
    noimg[noimg < 0] = noimg[noimg < 0] + np.dot(2, np.pi)
    noimg = np.dot(0.5, noimg)
    #end function smoothen_orientation_image
    return noimg

