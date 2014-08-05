from pylab import imread
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from fft_enhance_cubs import fft_enhance_cubs
from testfin import testfin
import pdb, math, threading, time
from test_bifurcation import test_bifurcation
from p import p
from dist2 import dist2
from tico import tico
from calc_orient import calc_orient
from trace_path import trace_path
from erode import erode

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

    enhimg = img.copy()
    
    [cimg1, o1, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o/4)
    [cimg, oimg2, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o/2)
    [cimg2, oimg, fimg, bwimg, eimg, enhimg] = fft_enhance_cubs(enhimg, blk_sz_o)
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

    #this loop takes 10 minutes
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
                if np.ceil(x/(float(blk_sz_c)/2)) <= cimg.shape[1] -1:
                    if np.ceil(y/(float(blk_sz_c)/2)) <= cimg.shape[0] -1:
                        cimg[np.ceil(y/(float(blk_sz_c)/2)), np.ceil(x/(float(blk_sz_c)/2)) - 1] = 255
                        cimg[np.ceil(y/(float(blk_sz_c)/2)), np.ceil(x/(float(blk_sz_c)/2))] = 255
                        cimg[np.ceil(y/(float(blk_sz_c)/2)) - 1, np.ceil(x/(float(blk_sz_c)/2))] = 255
                    else:
                        xpad = np.ceil(y/(float(blk_sz_c)/2))- (cimg.shape[0] -1)
                else:
                    ypad = np.ceil(x/(float(blk_sz_c)/2)) - (cimg.shape[1] -1)

    cimg = np.hstack((np.vstack((cimg, 255*np.ones((xpad,cimg.shape[1])))),np.vstack((255*np.ones((cimg.shape[0],ypad)),255*np.ones((xpad,ypad))))))
    cimg1[np.where(cimg1 < 0.51)[0]] = 255
    cimg[np.where(cimg < 0.51)[0]] = 255
    cimg2[np.where(cimg2 < 0.51)[0]] = 255

    inv_binim = (binim == 0)
    pdb.set_trace()
    thinned = erode(inv_binim)       #What is bwmorph?

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

    v = cimg.min(0)
    y = cimg.argmin(0) + 1
    dt1 = v.min(0)
    x = v.argmin(0) + 1
    

    delta1_y = y[x - 1]*float(blk_sz_c)/2
    delta1_x = x*float(blk_sz_c)/2

    v[x - 1] = 255
    v[x] = 255
    dt2 = v.min(0)
    x = v.argmin(0) + 1

    delta2_y = y[x - 1]*float(blk_sz_c)/2
    delta2_x = x*float(blk_sz_c)/2

    v[x - 1] = 255
    v[x] = 255
    dt3 = v.min(0)
    x = v.argmin(0) + 1

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

            test_m = np.min(o_img[0:y - yt, (np.max((x - 10), 1) -1):np.min(x + 10, o_img.shape[1])])
            
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
    pdb.set_trace() #to be deleted
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

    # Bibek's code starts here
    pdb.set_trace() #to be deleted
    # loop through image and find minutiae, ignore certain pixels for border
    for y in range(20, (img.shape[0] - 14 +1)):
        for x in range(21, (img.shape[1] - 21 +1)):
            if (thinned[(y -1), (x -1)] == 1):
                # only continue if pixel is white
                # calculate CN from Raymond Thai
                CN = 0
                sx = 0
                sy = 0
                for i in range(1, 9):
                    t1, x1, y1 = p(thinned, x, y, i) # nargout=3
                    t2, x2, y2 = p(thinned, x, y, i + 1) # nargout=3
                    CN = CN + abs(t1 - t2)
                CN = CN / 2


                if ((CN == 1) or (CN == 3)): #&& mask(y,x) > 0
                    skip = 0
                    for i in range(y - 5, (y + 5 +1)):
                        for j in range(x - 5, (x + 5 +1)):
                            if np.logical_and(np.logical_and(i > 0, j > 0), mask[(i -1), (j -1)]) == 0:
                                skip = 1

                    if skip == 1:
                        continue

                    t_a = np.array([])
                    c = 0
                    for e in range(y - 1, (y + 1 +1)):
                        for f in range(x - 1, (x + 1 +1)):
                            c = c + 1
                            np.insert(t_a,(c -1),orient_img_m[(e -1), (f -1)])

                    m_o = np.median(t_a) #orient_img_m(y,x);  #oimg(ceil(y/blk_sz_o),ceil(x/blk_sz_o));
                    m_f = 0 #fimg(ceil(y/blk_sz_o),ceil(x/blk_sz_o));
                    angle = m_o
                    if CN == 3:
                        CN, prog, sx, sy, ang = test_bifurcation(thinned, x, y, m_o, core_x, core_y) # nargout=5
                        if prog < 3:
                            continue
                        if ang < pi:
                            m_o = mod(m_o + pi, np.dot(2, pi))

                        #                  if ~(ang < pi && min(abs(m_o - ang),2*pi - abs(m_o - ang)) > abs(pi-abs(m_o - ang)))
                        #                     m_o = mod(m_o+pi,2*pi);
                        #                  end
                        print 'y = ' + str(y) +'\n'
                        print 'x = ' + str(x) +'\n'
                        print 'm_o = ' + str(m_o) +'\n'
                        print 'ang = ' + str(ang) +'\n'
                        print 'sy = ' + str(sy) +'\n'
                        print 'sx = ' + str(sx) +'\n'
                        print np.min(abs(m_o - ang), np.dot(2, pi) - abs(m_o - ang))
                    else:
                        progress = 0
                        xx = x
                        yy = y
                        pao = - 1
                        pos = 0
                        while progress < 15 and xx > 1 and yy > 1 and yy < img.shape[0] and xx < img.shape[1] and pos > - 1:
                            pos = - 1
                            for g in range(1, 9):
                                ta, xa, ya = p(thinned, xx, yy, g) # nargout=3
                                tb, xb, yb = p(thinned, xx, yy, g + 1) # nargout=3
                                if (ta > tb) and pos == - 1 and g != pao:
                                    pos = ta
                                    if g < 5:
                                        pao = 4 + g
                                    else:
                                        pao = mod(4 + g, 9) + 1
                                    xx = xa
                                    yy = ya
                            progress = progress + 1

                        if progress < 10:
                            continue
                        if mod(atan2(y - yy, xx - x), np.dot(2, pi)) > pi:
                            m_o = m_o + pi
                    minutiae[(minu_count -1), :] = np.array([x, y, CN, m_o, m_f, 1]).reshape(1, -1)
                    min_path_index[(minu_count -1), :] = np.array([sx, sy])
                    minu_count = minu_count + 1
            #end of if loop if pixel white
        # enf of for y
    # end of for x

    # Identify the ellipse enclosed by the largest rectangle spanned by the 
    # set of initially detected minutiaes (4/8/2006 - Paul Kwan)
    min_x = np.min(minutiae[:, 0])
    max_x = np.max(minutiae[:, 0])
    min_y = np.min(minutiae[:, 1]) / 1.1
    max_y = np.dot(np.max(minutiae[:, 1]), 1.1)
    mid_x = (min_x + max_x) / 2
    mid_y = (min_y + max_y) / 2

    if (max_x - min_x) > (max_y - min_y):         # major axis is on the x-axis
        major_axis_len = max_x - min_x         # Refer to figure
        M_Y1 = (max_y - min_y) / 2
        F1_Y1 = major_axis_len / 2
        M_F1 = sqrt(F1_Y1 ** 2 - M_Y1 ** 2)        # Use Pythagoras' theorem
        F1_x = mid_x - M_F1
        F1_y = mid_y
        F2_x = mid_x + M_F1
        F2_y = mid_y
    else:
        major_axis_len = max_y - min_y        # Refer to figure
        M_X1 = (max_x - min_x) / 2
        F1_X1 = major_axis_len / 2
        M_F1 = sqrt(F1_X1 ** 2 - M_X1 ** 2)         # Use Pythagoras' theorem
        F1_x = mid_x
        F1_y = mid_y + M_F1
        F2_x = mid_x
        F2_y = mid_y - M_F1

    clean_list = np.array([])
    c_index = 1
    thinned_old = thinned.copy()
    minu_count = minu_count - 1
    
    # Filter potential false endings by excluding those outside of the 
    # elliptical region
    t_minutiae = np.array([])
    t_minu_count = 1
    t_mpi = np.array([])
    pre_minu_count1 = minu_count


    for i in range(1, (minu_count +1)):
        X = minutiae[(i -1), 0]
        Y = minutiae[(i -1), 1]
        rc = 0
        for y in range(max(Y - 2, 1), (min(Y + 2, binim.shape[0]) +1)):
            if rc > 0:
                break
            for x in range(max(X - 2, 1), (min(X + 2, binim.shape[1]) +1)):
                if mask[(y -1), (x -1)] == 0:
                    rc = rc + 1
                    break
        if rc > 0:
            continue
        else:
            np.insert(t_minutiae, (t_minu_count -1), minutiae[(i -1), :])
            np.insert(t_mpi, (t_minu_count -1), min_path_index[(i -1), :])
            t_minu_count = t_minu_count + 1

    minutiae = t_minutiae.copy()
    min_path_index = t_mpi.copy()

    minu_count = minutiae.shape[0]

    t_minu_count = 1
    t_minutiae = np.array([])

    pre_minu_count2 = minu_count
    print 'pre_minu_count2 =' + str(pre_minu_count2) + '\n'
    
    dist_m = dist2(minutiae[:, 0:2], minutiae[:, 0:2])
    #dist_m(find(dist_m == 0)) == 1000;

    dist_test = 49
    #if minu_count > 50
    #   dist_test=81;
    #end

    for i in range(1, (minu_count +1)):
        reject_flag = 0
        t_a = minutiae[(i -1), 3]
        P_x = minutiae[(i -1), 0]
        P_y = minutiae[(i -1), 1]

        for j in range(i + 1, (minu_count +1)):
            if dist_m[(i -1), (j -1)] <= dist_test:
                reject_flag = 1

        if reject_flag == 0 and mask[(P_y -1), (P_x -1)] > 0:
            reverse_p = 0
            if min_path_index[(i -1), 0] == 0:
                x = P_x
                y = P_y
            else:
                x = min_path_index[(i -1), 0]
                y = min_path_index[(i -1), 1]

            iter_ = 0
            p1x = P_x
            p1y = P_y
            p2x = P_x
            p2y = P_y
            minutiae_o_change = np.array([])

            x1 = x
            y1 = y
            xs = x
            ys = y
            p_uv = np.array([(x / math.sqrt(x ** 2 + y ** 2)), (y / math.sqrt(x ** 2 + y ** 2))])
            iter_ = 0

            for m in range(1, (path_len +1)):
                iter_ = iter_ + 1
                cn = 0

                for ii in range(1, 9):
                    t1, x_A, y_A = p(thinned, x1, y1, ii) # nargout=3
                    t2, x_B, y_B = p(thinned, x1, y1, ii + 1) # nargout=3
                    cn = cn + abs(t1 - t2)
                cn = cn / 2

                if (cn != 3 and cn != 4) or m == 1: #&& mask(x1,y1) > 0
                    for n in range(1, 9):
                        if reverse_p == 0 or iter_ > 1:
                            ta, xa, ya = p(thinned, x1, y1, n) # nargout=3
                        else:
                            ta, xa, ya = p(thinned, x1, y1, 9 - n) # nargout=3
                        if ta == 1 and (xa != p1x or ya != p1y) and (xa != x or ya != y):
                            p1x = x1
                            p1y = y1
                            x1 = xa
                            y1 = ya
                            break

                if np.mod(iter_, sample_intv) == 0:
                    if xs == x1 and ys == y1:
                        minutiae_o_change = np.insert(minutiae_o_change, (iter_ / sample_intv -1), - 1)
                    else:
                        norm_s = math.sqrt((x1 - xs) ** 2 + (y1 - ys) ** 2)
                        xv = (x1 - xs) / norm_s
                        yv = (y1 - ys) / norm_s
                        ptx = x - xs
                        pty = y - ys
                        norm_p = math.sqrt((ptx) ** 2 + (pty) ** 2)
                        ptx = ptx / norm_p
                        pty = pty / norm_p

                        #inner product
                        i_prod = (np.dot(xv, ptx) + np.dot(yv, pty))
                        minutiae_o_change = np.insert(minutiae_o_change,(iter_ / sample_intv -1), math.acos(i_prod))
                        xs = x1
                        ys = y1
                        
            minutiae_o_change = np.insert(minutiae_o_change, 0, 1)

            #     [temp_o1, temp_r1, mx1, my1] = tico(binim, P_x, P_y, oimg, orient_img_m, o_rel, [12 24 36 48 60 72 84] , [14 18 24 28 30 34 28]   , 1, minutiae(i,4), blk_sz_o);
            temp_o1, temp_r1, mx1, my1 = tico(binim, P_x, P_y, oimg, orient_img_m, o_rel, np.array([12, 24, 36, 48, 60, 72, 84]), np.array([14, 18, 24, 28, 30, 34, 28]), 0, - 1, blk_sz_o) # nargout=4
            t_minutiae = np.insert(t_minutiae, (t_minu_count -1), minutiae[(i -1), :])
            min_o_change[(t_minu_count -1), :] = minutiae_o_change
            orientations = np.insert(orientations, (t_minu_count -1), temp_o1)
            radii = np.insert(radii, (t_minu_count -1), temp_r1)
            t_minu_count = t_minu_count + 1

    minutiae = t_minutiae.copy()
    minu_count = t_minu_count - 1


    tmpvec1 = img.shape[0] * np.ones(shape=(minu_count, 1), dtype='float64')
    tmpvec2 = np.ones(shape=(minu_count, 1), dtype='float64')
    minutiae_for_sc = np.hstack((minutiae[:, 0] / img.shape[1], (tmpvec1 - minutiae[:, 1] + tmpvec2) / img.shape[0]))
    dist_m = math.sqrt(dist2(minutiae_for_sc[:, 0:2], minutiae_for_sc[:, 0:2]))

    neighbours = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    n_i = 1

    for i in range(1, (minu_count +1)):
        t_a = minutiae[(i -1), 3]
        P_x = minutiae[(i -1), 0]
        P_y = minutiae[(i -1), 1]

        d = np.sort(dist_m[(i -1), :]) # nargout=2
        ind = np.argsort(dist_m[(i -1), :])
        for j in range(1, (minu_count +1)):
            if dist_m[(i -1), (ind[(j -1)] -1)] == 0:
                continue
            diff_x = minutiae[(i -1), 0] - minutiae[(ind[(j -1)] -1), 0]
            diff_y = minutiae[(i -1), 1] - minutiae[(ind[(j -1)] -1), 1]
            angle = np.mod(math.atan2(diff_y, diff_x) + np.dot(2, pi), np.dot(2, pi))
            theta = np.mod(minutiae[(i -1), 3] - angle + np.dot(2, pi), np.dot(2, pi))

            theta_t = np.mod(math.atan2(minutiae[(i -1), 1] - minutiae[(ind[(j -1)] -1), 1], minutiae[(i -1), 0] - minutiae[(ind[(j -1)] -1), 0]), np.dot(2, pi))

            ridge_count = 0
            p_y = minutiae[(i -1), 1]
            p_x = minutiae[(i -1), 0]
            t_x = 0
            t_y = 0
            current = 1
            radius = 1

            while p_y != minutiae[(ind[(j -1)] -1), 1]:
                if thinned[(p_y -1), (p_x -1)] > 0 and current == 0 and (t_x != p_x or t_y != p_y):
                    current = 1
                    ridge_count = ridge_count + 1
                else:
                    if thinned[(p_y -1), (p_x -1)] == 0:
                        current = 0
                t_x = p_x
                t_y = p_y
                p_x = np.around(minutiae[(i -1), 0] - np.dot(radius, math.cos(theta_t)))
                p_y = np.around(minutiae[(i -1), 1] - np.dot(radius, math.sin(theta_t)))
                radius = radius + 1

            co = calc_orient(orientations[(i -1), :], radii[(i -1), :], orientations[(ind[(j -1)] -1), :], radii[(ind[(j -1)] -1), :])
            neighbours[(n_i -1), :] = np.array([i, ind[(j -1)], dist_m[(i -1), (ind[(j -1)] -1)], ridge_count, theta_t, minutiae[(ind[(j -1)] -1), 2], minutiae[(ind[(j -1)] -1), 4], co, np.min(abs(minutiae[(i -1), 3] - minutiae[(ind[(j -1)] -1), 3]), np.dot(2, pi) - abs(minutiae[(i -1), 3] - minutiae[(ind[(j -1)] -1), 3]))])
            n_i = n_i + 1

    pre_singular_min = minutiae

    if core_val < 1:
        minutiae = np.insert(minutiae, (minu_count + 1 -1), np.array([core_x, core_y, 5, start_t, 0, 1]))
        minu_count = minu_count + 1

    if dt1 < 1:
        minutiae = np.insert(minutiae, (minu_count + 1 -1), np.array([delta1_x, delta1_y, 7, 0, 1, 1]))
        minu_count = minu_count + 1

    if dt2 < 1:
        minutiae = np.insert(minutiae, (minu_count + 1 -1), np.array([delta2_x, delta2_y, 7, 0, 1, 1]))
        minu_count = minu_count + 1

    if dt3 < 1:
        minutiae = np.insert(minutiae, (minu_count + 1 -1), np.array([delta3_x, delta3_y, 7, 0, 1, 1]))
        minu_count = minu_count + 1

    tmpvec1 = img.shape[0] * np.ones(shape=(minu_count, 1), dtype='float64')     #JI: m array of 1's multiplied by image vertical size 
    tmpvec2 = np.ones(shape=(minu_count, 1), dtype='float64')     #JI: m array of 1's 

    minutiae_for_sc = np.hstack((minutiae[:, 0] / img.shape[1], (tmpvec1 - minutiae[:, 1] + tmpvec2) / img.shape[0]))    #convert to x-y cartesian co-ordinates in [0,1]

    img_sc = trace_path(thinned, pre_singular_min, 20)

    # make minutiae image
    if display_flag == 1:
        minutiae_img = np.zeros(shape=(img.shape[0], img.shape[1], 3), dtype='uint8')
        for i in range(1, (minu_count +1)):
            x1 = minutiae[(i -1), 0]
            y1 = minutiae[(i -1), 1]

            if minutiae[(i -1), 2] == 1:
                if minutiae[(i -1), 3] > pi:
                    for k in range(y1 - 2, (y1 + 2 +1)):
                        for l in range(x1 - 2, (x1 + 2 +1)):
                            minutiae_img[(k -1), (l -1), :] = np.array([255, 0, 0]).reshape(1, -1)
                else:
                    for k in range(y1 - 2, (y1 + 2 +1)):
                        for l in range(x1 - 2, (x1 + 2 +1)):
                            minutiae_img[(k -1), (l -1), :] = np.array([205, 100, 100]).reshape(1, -1)

            elif minutiae[(i -1), 2] == 2:
                for k in range(y1 - 2, (y1 + 2 +1)):
                    for l in range(x1 - 2, (x1 + 2 +1)):
                        minutiae_img[(k -1), (l -1), :] = np.array([255, 0, 255]).reshape(1, -1)

            elif minutiae[(i -1), 2] == 3:
                if minutiae[(i -1), 3] > pi:
                    for k in range(y1 - 2, (y1 + 2 +1)):
                        for l in range(x1 - 2, (x1 + 2 +1)):
                            minutiae_img[(k -1), (l -1), :] = np.array([0, 0, 255]).reshape(1, -1)
                else:
                    for k in range(y1 - 2, (y1 + 2 +1)):
                        for l in range(x1 - 2, (x1 + 2 +1)):
                            minutiae_img[(k -1), (l -1), :] = np.array([255, 0, 255]).reshape(1, -1)

            elif minutiae[(i -1), 2] == 5:
                for k in range(y1 - 4, (y1 + 4 +1)):
                    for l in range(x1 - 4, (x1 + 4 +1)):
                        minutiae_img[(k -1), (l -1), :] = np.array([0, 255, 0]).reshape(1, -1)
            elif minutiae[(i -1), 2] > 5:
                for k in range(y1 - 2, (y1 + 2 +1)):
                    for l in range(x1 - 2, (x1 + 2 +1)):
                        minutiae_img[(k -1), (l -1), :] = np.array([128, 128, 0]).reshape(1, -1) # gold for delta

        # merge thinned image and minutia_img together
        combined = minutiae_img.astype('uint8')
        for x in range(1, (binim.shape[1] +1)):
            for y in range(1, (binim.shape[0] +1)):
                if mask[(y -1), (x -1)] == 0:
                    combined[(y -1), (x -1), :] = np.array([0, 0, 0]).reshape(1, -1)
                    continue
                if (thinned[(y -1), (x -1)]):                     # binim(y,x))
                    combined[(y -1), (x -1), :] = np.array([255, 255, 255]).reshape(1, -1)
                else:
                    combined[(y -1), (x -1), :] = np.array([0, 0, 0]).reshape(1, -1)
                # end if
                if ((minutiae_img[(y -1), (x -1), 2] != 0) or (minutiae_img[(y -1), (x -1), 0] != 0)) or (minutiae_img[(y -1), (x -1), 1] != 0):
                    combined[(y -1), (x -1), :] = minutiae_img[(y -1), (x -1), :].copy()
           
        if core_val < 1 and YA > 0:
            for k in range(YA - 2, (YA + 2 +1)):
                for l in range(XA - 2, (XA + 2 +1)):
                    combined[(k -1), (l -1), :] = np.array([20, 255, 250]).reshape(1, -1)
            for k in range(YB - 2, (YB + 2 +1)):
                for l in range(XB - 2, (XB + 2 +1)):
                    combined[(k -1), (l -1), :] = np.array([20, 255, 250]).reshape(1, -1)
        plt.figure(1)
        #subplot(2,2,1), subimage(img), title(filename)
        #subplot(2,3,2), subimage(cimg), title('orientation image 0')
        #subplot(2,3,3), subimage(cimg1), title('orientation image 1')
        #subplot(2,2,1), subimage(minutiae_img), title('minutiae points')
        plt.subplot(1, 2, 1)
        plt.subimage(combined)
        plt.title(filename)
        plt.subplot(1, 2, 2)
        plt.subimage(thinned)
        plt.title('thinned image')
        # subplot(2,2,3), subimage(o_img), title('orientation image (sine)')
        #  subplot(2,3,5), subimage(eimg), title('f image ')
        # subplot(2,2,4), subimage(img_sc), title('orientation image ')
        plt.show()
        #plotridgeorient(orient_img, 20, img, 2)
    print minutiae.shape + '\n'
    print minutiae_for_sc.shape + '\n'
    print 'minutiae=' + str(minutiae) + '\n'
    
    
    
###########TESTING############
extract_finger('DB1_B\\108_3.tif')

##thread = threading.Thread()
##thread.run = main
##
##manager = plt.get_current_fig_manager()
##manager.window.after(100, thread.start)
##plt.figure(1)
##plt.show()
