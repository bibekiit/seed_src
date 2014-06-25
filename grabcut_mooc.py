#!/usr/bin/env python
'''
===============================================================================
Interactive Image Segmentation using GrabCut algorithm.

This sample shows interactive image segmentation using grabcut algorithm.

USAGE :
    python grabcut.py <filename>

README FIRST:    
    Two windows will show up, one for input and one for output.
    
    At first, in input window, draw a rectangle around the object using 
mouse right button. Then press 'n' to segment the object (once or a few times)
For any finer touch-ups, you can press any of the keys below and draw lines on 
the areas you want. Then again press 'n' for updating the output.

Key 'n' - To update the segmentation
Key 'r' - To reset the setup
Key 's' - To save the results
===============================================================================
'''

import numpy as np
import cv2
import sys

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}

# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness

def onmouse(event,x,y,flags,param):
    global img,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
    
    # Draw Rectangle
    if event == cv2.EVENT_RBUTTONDOWN:
        rectangle = True
        ix,iy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if rectangle == True:
            img = img2.copy()
            cv2.rectangle(img,(ix,iy),(x,y),GREEN,2)
            rect = (ix,iy,abs(ix-x),abs(iy-y))
            rect_or_mask = 0

    elif event == cv2.EVENT_RBUTTONUP:
        rectangle = False
        rect_over = True
        cv2.rectangle(img,(ix,iy),(x,y),BLUE,2)
        print (ix,iy),(x,y)
        rect = (ix,iy,abs(ix-x),abs(iy-y))
        rect_or_mask = 0
        print " Now press the key 'n' a few times until no further change \n"
        
    # draw touchup curves
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if rect_over == False:
            print "first draw rectangle \n"
        else:
            drawing = True
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing == True:
            drawing = False
            cv2.circle(img,(x,y),thickness,value['color'],-1)
            cv2.circle(mask,(x,y),thickness,value['val'],-1)
        
# print documentation
print __doc__

# Loading images
if len(sys.argv) == 2:
    filename = sys.argv[1] # for drawing purposes
else:
    print "No input image given, so loading default image, lena.jpg \n"
    print "Correct Usage : python grabcut.py <filename> \n"
    filename = "C:\\Users\\Bibek\\Downloads\\images_of_fingerprint\\test_1\\019_7.jpg"

img = cv2.imread(filename);
img2 = img.copy()                               # a copy of original image
mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
output = np.zeros(img.shape,np.uint8)           # output image to be shown

# input and output windows
cv2.namedWindow('output')
cv2.namedWindow('input')
cv2.setMouseCallback('input',onmouse)
cv2.moveWindow('input',img.shape[1]+10,90)

print " Instructions : \n"
print " Draw a rectangle around the object using right mouse button \n"

while(1):
    cv2.imshow('output',output)
    cv2.imshow('input',img)
    k = 0xFF & cv2.waitKey(1)

    if k == 27:         # esc to exit
        break
    elif k == ord('r'): # reset everything
        print "resetting \n"
        rect = (0,0,1,1)
        drawing = False         
        rectangle = False       
        rect_or_mask = 100 
        rect_over = False     
        value = DRAW_FG         
        img = img2.copy()
        mask = np.zeros(img.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
        output = np.zeros(img.shape,np.uint8)           # output image to be shown
    
    elif k == ord('n'): # segment the image
        
        if (rect_or_mask == 0):         # grabcut with rect
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64)    
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
            rect_or_mask = 1
        elif rect_or_mask == 1:         # grabcut with mask
            bgdmodel = np.zeros((1,65),np.float64)
            fgdmodel = np.zeros((1,65),np.float64) 
            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
        print rect, rect[0], rect[1], rect[2], rect[3]
    elif k == ord('s'): # save image
        #bar = np.zeros((img.shape[0],5,3),np.uint8)
        #res = np.hstack((img2,bar,img,bar,output))
        croppedImage = output[rect[1]:(rect[1]+rect[3]), rect[0]:(rect[0]+rect[2])]
        cv2.imshow('output2', croppedImage)
        cv2.imwrite('C:\\Users\\Bibek\\Downloads\\images_of_fingerprint\\test_2\\temp.png', croppedImage)
        print " Result saved as image \n"
        
            

    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
    cv2.imshow('mask2',mask2)
    output = cv2.bitwise_and(img2,img2,mask=mask2)   

cv2.destroyAllWindows()
