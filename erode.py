import cv2
import numpy as np
import matplotlib.pyplot as plt
import pdb

def erode(img):
    pdb.set_trace()
    img = img.astype('int')
    plt.imshow(img, cmap = plt.get_cmap("gray"))
    plt.show()
    size = np.size(img)
    skel = np.zeros(img.shape,np.uint8)
     
    ret,img = cv2.threshold(img,20,np.max(img),0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    done = False
     
    while( not done):
        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)
        skel = cv2.bitwise_or(skel,temp)
        img = eroded.copy()
     
        zeros = size - cv2.countNonZero(img)
        if zeros==size:
            done = True
 
    cv2.imshow("skel",skel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return skel

############TESTING#############
erode(cv2.imread('inv_binim.png',cv2.CV_LOAD_IMAGE_GRAYSCALE))
