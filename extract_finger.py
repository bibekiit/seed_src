from pylab import imread
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from fft_enhance_cubs import fft_enhance_cubs
from testfin import testfin
import pdb

def rgb2gray(rgb):
    """Numpy does not have a built-in function to convert from rgb to gray.
    Reference for the present method is http://stackoverflow.com/questions/12201577/convert-rgb-image-to-grayscale-in-python"""
    return np.dot(rgb[...,:3],[0.299,0.587, 0.144])





def extract_finger(filename):
    """Yet to be written"""

    blk_sz_c = 24
    blk_sz_o = 24

    img = mpimg.imread(filename)

    #convert to gray scale

    if img.ndim == 3:   #colour image
        img = rgb2gray(img)

    yt = 1
    yb = img.shape[1]
    xr = img.shape[0]
    xl = 1

    YA = 0
    YB = 0
    XA = 0
    XB = 0

    delta_test = 0

    for x in range(55):
        if np.where(img[x,:] < 200)[0].size < 8:
            img[0:x, :] = 255
            yt = x + 1


    for x in range(224, img.shape[0]):
        if np.where(img[x,:] < 200)[0].size < 3:
            img[x - 17:img.shape[0],:] = 255
            yb = x + 1
            break


    for y in range(199, img.shape[1]):
        if np.where(img[:,y] < 200)[0].size < 1:
            img[:,y:img.shape[1]] = 255
            xr = y + 1
            break


    for y in range(75):
        if np.where(img[:,y] < 200)[0].size < 1:
            img[:,0:y] = 255
            xl = y + 1

    pdb.set_trace()
    enhimg = img.copy()

##    [cimg1, o1, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o/4)
##    plt.imshow(enhimg, cmap = plt.get_cmap("gray"))
##    plt.show()
##    [cimg, oimg2, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o/2)
##    plt.imshow(enhimg, cmap = plt.get_cmap("gray"))
##    plt.show()
##    [cimg2, oimg, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o)
##    plt.imshow(enhimg, cmap = plt.get_cmap("gray"))
##    plt.show()
    [newim, binim, mask, o_rel, orient_img] = testfin(enhimg)       #testfin is from Dr. Peter Kovesi's code
    [cimg, oimg2, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(img, -1)
    [newim, binim, mask, o_rel, orient_img_m] = testfin(enhimg)

    theta1 = 0.5

    mask_t = mask.copy()

    for y in range(19, mask.shape[0] - blk_sz_c*2 + 1):
        for x in range(blk_sz_c, mask.shape[1] - blk_sz_c*2 + 1):
            n_mask = 0
            for yy in range(-1,2):
                for xx in range(-1,2):
                    y_t = y + yy*blk_sz_c
                    x_t = x + xx*blk_sz_c
                    if y_t > 0 and x_t > 0 and (y_t != y or x_t != x) and mask[y_t - 1, x_t - 1] == 0:
                        n_mask = n_mask + 1


            if n_mask == 0:
                continue

            if mask[y - 1, x - 1] == 0 or y > mask.shape[0] - 20 or y < yt or y > yb or x < xl or x > xr:
                cimg2[np.ceil(y/float(blk_sz_c)) - 1, np.ceil(x/float(blk_sz_c)) - 1] = 255
                mask_t[y - 1, x - 1] = 0
                continue

            for i in range(y, y + 2):
                for j in range(x - 9, x + 10):
                    if i > 0 and j > 0 and i < mask.shape[0] and j < mask.shape[1] and mask[i - 1, j - 1] > 0:
                        continue                        
                    else:
                        cimg2[np.ceil(y/float(blk_sz_c)) - 1, np.ceil(x/float(blk_sz_c)) - 1] = 255
                        mask_t[y - 1, x - 1] = 0
                        break


    mask = mask_t.copy()

    cimg1[np.where(cimg1 > 0.9)[0]] = 255
    cimg1[np.where(cimg > 0.9)[0]] = 255
    cimg2[np.where(cimg2 > 0.875)[0]] = 255

    for y in range(1, mask.shape[0] - blk_sz_c*2 + 1):
        for x in range(blk_sz_c, mask.shape[1] - blk_sz_c*2 + 1):
            for i in range(y - 25, y + 26):
                for j in range(x - 25, x + 26):
                    if i > 0 and j > 0 and i < mask.shape[0] and j < mask.shape[1] and mask[i - 1, j - 1] > 0:
                        continue
                    else:
                        cimg2[np.ceil(y/float(blk_sz_c)) - 1, np.ceil(x/float(blk_sz_c)) - 1] = 255
                        break

    for y in range(1, mask.shape[0] - blk_sz_c*2 + 1):
        for x in range(1, mask.shape[1] - blk_sz_c + 1):
            if cimg[np.ceil(y/(float(blk_sz_c)/2)) - 1, np.ceil(x/(float(blk_sz_c)/2)) - 1] == 255:
                cimg1[np.ceil(y/(float(blk_sz_c)/4)) - 1, np.ceil(x/(float(blk_sz_c)/4)) - 1] = 255
                cimg1[np.ceil(y/(float(blk_sz_c)/4)), np.ceil(x/(float(blk_sz_c)/4)) - 1] = 255
                cimg1[np.ceil(y/(float(blk_sz_c)/4)) - 1, np.ceil(x/(float(blk_sz_c)/4))] = 255
                cimg1[np.ceil(y/(float(blk_sz_c)/4)), np.ceil(x/(float(blk_sz_c)/4))] = 255


    

    for y in range(1, mask.shape[0] - blk_sz_c + 1):
        for x in range(1, mask.shape[1] - blk_sz_c + 1):
            if cimg2[np.ceil(y/float(blk_sz_c)) - 1, np.ceil(x/float(blk_sz_c)) - 1] == 255:
                cimg[np.ceil(y/(float(blk_sz_c)/2)) - 1, np.ceil(x/(float(blk_sz_c)/2)) - 1] = 255
                cimg[np.ceil(y/(float(blk_sz_c)/2)), np.ceil(x/(float(blk_sz_c)/2)) - 1] = 255
                cimg[np.ceil(y/(float(blk_sz_c)/2)) - 1, np.ceil(x/(float(blk_sz_c)/2))] = 255
                cimg[np.ceil(y/(float(blk_sz_c)/2)), np.ceil(x/(float(blk_sz_c)/2))] = 255


    cimg1[np.where(cimg1 < 0.51)[0]] = 255
    cimg[np.where(cimg < 0.51)[0]] = 255
    cimg2[np.where(cimg2 < 0.51)[0]] = 255

    inv_binim = (binim == 0)
    thinned = bwmorph(inv_binim, 'thin', Inf)       #What is bwmorph?

    mask_t = mask.copy()

    if np.where(mask[124:150, 149:250] > 0)[0].size > 0 and np.where(mask[249:275, 149:250] > 0)[0].size > 0:
        mask[149:250, 149:250] = 1

    method = -1
    core_path = 255
    core_y = 0
    core_x = 0
    core_val = 0
    lc = 0
    o_img = np.sin(orient_img)
    o_img[np.where(mask == 0)[0]] = 1

    lower_t = 0.1

    #Line 202 how to code
    #Line 203 how to code

    delta1_y = y[x - 1]*float(blk_sz_c)/2
    delta1_x = x*float(blk_sz_c)/2

    v[x - 1] = 255
    v[x] = 255
    #Line 209, how to code?
    delta2_y = y[x - 1]*float(blk_sz_c)/2
    delta2_x = x*float(blk_sz_c)/2

    v[x - 1] = 255
    v[x] = 255
    #Line 215 how to code
    delta3_y = y[x - 1]*float(blk_sz_c)/2
    delta3_x = x*float(blk_sz_c)/2

    db = 60

    if dt1 < 1 and delta1_y + db < core_y and delta1_y > 15 or dt2 < 1 and delta2_y + db < core_y and delta2_y > 15 or dt3 < 1 and delta3_y + db < core_y \
       and delta3_y > 15:
        core_val = 255

    for y in range(10, o_img.shape[0] - 9):
        for x in range(10, o_img.shape[1] - 9):

            s1 = 0
            t = 10

            if y < 50 and x > 250:
                t = 11

            if y > 38:
                yt = 20
            else:
                yt = 5

            if lc > 0.41 and (core_y + 60 < y):
                break

            if mask[y - 1, x - 1] == 0 or mask[max(y - t,1) - 1, x - 1] == 0 or mask[y - 1, min(x + t, o_img.shape[1]) - 1] == 0 \
               or mask[y - 1, max(x - t, 1) - 1] == 0 or mask[max(y - t, 1) - 1, min(x + t, o_img.shape[1]) - 1] == 0 \
               and mask[max(y - t, 1) - 1, max(x - t, 1) - 1] == 0 or o_img[y - 1, x - 1] < lc or o_img[y - 1, x - 1] < 0.1:
                continue
                
               

            if dt1 < 1 and delta_y + db < y and delta_y > 15 or dt2 < 1 and delta2_y + db < y and delta2_y > 15 or \
               dt3 < 1 and delta3_y + db < y and delta3_y > 15:
                continue

            #Line 255 how to code?

            if test_m.shape > 0 and np.min(test_m) >= 0.17:
                continue

            for a in range(y, y + 3):
                for b in range(x, x + 2):
                    s1 = s1 + o_img[a - 1, b - 1]

            s1 = float(s1)/6

            s2 = []
            i = 1
            
            for a in range(y - 3, y):
                for b in range(x, x + 2):
                    s2[i - 1] = o_img[a - 1, b - 1]
                    i = i + 1


            if min(s2) < lower_t:
                s2 = sum(s2)/6
            else:
                s2 = s1     #s2 was a list, s1 is a no., this statement makes s2 a number.

            s3 = []
            i = 1
            for a in range(y, y + 3):
                for b in range(x + 2, x + 4):
                    s3[i - 1] = o_img[a - 1, b - 1]
                    i = i + 1

            if min(s3) < lower_t:
                s3 = sum(s3)/float(6)
            else:
                s3 = s1

            s4 = []
            i = 1
            for a in range(y, y + 3):
                for b in range(x - 2, x):
                    s4[i - 1] = o_img[a - 1, b - 1]
                    i = i + 1

            if min(s4) < lower_t:
                s4 = sum(s4)/float(6)
            else:
                s4 = s1

            s5 = []
            i = 1
            for a in range(y - 3, y):
                for b in range(x - 2, x):
                    s5[i - 1] = o_img[a - 1, b - 1]
                    i = i + 1

            if min(s5) < lower_t:
                s5 = sum(s5)/float(6)
            else:
                s5 = s1

            s6 = []
            for a in range(y - 3, y):
                for b in  range(x + 2, x + 4):
                    s6[i - 1] = o_img[a - 1, b - 1]
                    i = i + 1

            if min(s6) < lower_t:
                s6 = sum(s6)/float(6)
            else:
                s6 = s1

            if s1 - s2 > core_val:
                core_val = s1 - s2
                core_x = x
                core_y = y
                lc = o_img[y - 1, x - 1]
                method = 1

            if s1 - s3 > core_val:
                core_val = s1 - s3
                core_x = x
                core_y = y
                lc = o_img[y - 1, x - 1]
                method = 2

            if x < 300 and s1 - s4 > core_val:
                core_val = s1 - s4
                core_x = x
                core_y = y
                lc = o_img[y - 1, x - 1]
                method = 3

            if x < 300 and s1 - s5 > core_val:
                core_val = s1 - s5
                core_x = x
                core_y = y
                lc = o_img[y - 1, x - 1]
                method = 4

            if s1 - s6 > core_val:
                core_val = s1 - s6
                core_x = x
                core_y = y
                lc = o_img[y - 1, x - 1]
                method = 5

    #nested for loop ends here

    yt = 0

    if core_y > 37:
        yt = 20
    else:
        yt = 5
    #Printing is as desired by MATLAB code, reason for printing not known.
    print "lc = ", lc
    print "core_y = ", core_y
    print "core_x = ", core_x
    print "method = ", method
    test_smooth = 100

    if core_y > 0:
        test_smooth = np.sum(np.sum(o_img[core_y - yt - 6:core_y - yt + 5, core_x - 6: core_x + 5],0))

    if lc > 0.41 and (test_smooth < 109.5 and method != 2 or test_smooth < 100):
        start_t = 0
        core_val = float(1)/(core_val + 1)
        print "core_x = ", core_x
        print "core_y = ", core_y
        #How to code line 404?
    else:
        core_x = 0
        core_y = 0
        core_val = 255

    mask = mask_t.copy()
    display_flag = 1

    ocodes = []
    orientations = []
    radii = []

    sample_intv = 5
    path_len = 45
    min_o_change = np.zeros([1, floor(float(path_len)/sample_intv)])
    #to append more rows in min_o_change, we'll use np.vstack() function

    minu_count = 1
    minutiae = np.array([0,0,0,0,0,1])

    #we've finished 424 lines
    
    
extract_finger('DB1_B\\108_2.tif')
    
