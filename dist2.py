import numpy as np

def dist2(x,c):

    (ndata,dimx) = x.shape
    (ncentres,dimc) = c.shape

    if dimx != dimc:
        raise TypeError('Data dimension does not match dimension of centres')

    B = np.zeros([ndata,ncentres])

    for i in range(ndata):
        for j in range(ncentres):
            B[i,j] = sum(np.square(x[i,:] - c[j,:]))


    return B
            
