import scipy.ndimage.filters as snf
import numpy as np
def filter2(x, b):
    [mx,nx] = x.shape
    stencil = np.rot90(b,2)
    [ms,ns] = stencil.shape
    y = snf.convolve(x,stencil)
    return y
