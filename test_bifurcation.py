import numpy as np
import math, time

from p import p
from dist2 import dist2



def test_bifurcation(img,x,y,o,core_x,core_y):

    progress = 1
    path_len = 4

    pax = 0
    pay = 0
    pbx = 0
    pby = 0
    pcx = 0
    pcy = 0

    pao = 0
    pbo = 0
    pco = 0

    for i in  range(1,9):

        [ta, xa, ya] = p(img, x, y, i)
        [tb, xb, yb] = p(img, x, y, i + 1)

        if ta > tb:
            if pao == 0:
                if i < 5:
                    pao = 4 + i
                else:
                    pao = np.mod(4 + i, 9) + 1

                pax = xa
                pay = ya

            else:
                if pbo == 0:
                    if i < 5:
                        pbo = 4 + i
                    else:
                        pbo = np.mod(4 + i,9) + 1

                    pbx = xa
                    pby = ya

                else:
                    if i < 5:
                        pco = 4 + i
                    else:
                        pco = np.mod(4 + i, 9) + 1

                    pcx = xa
                    pcy = ya

                    break

    xaa = 0
    yaa = 0
    xbb = 0
    ybb = 0
    xcc = 0
    ycc = 0

    hist = np.array([[pay,pax],[pby, pbx], [pcy,pcx],[y,x]])
    stop = False

    while progress < path_len and not False:
        progress = progress + 1
        da = 0
        db = 0
        dc = 0

        if pao != 0:
            i = 1
            cn = 0

            for ii in range(1,9):
                [t1, x_A, y_A] = p(img, pax, pay, ii)
                [t2, x_B, y_B] = p(img, pax, pay, ii + 1)
                cn = cn + abs(t1 - t2)

            cn = cn/2.0

            if cn == 1 or cn == 3:
                stop = True

            while i < 9 and da == 0:

                [ta, xa, ya] = p(img, pax, pay, i)
                [tz, xz, yz] = p(img, pax, pay, i + 1)

                ind_y = np.where(hist[:,0] == ya)[0]
                ind_x = np.where(hist[ind_y,1] == xa)[0]

                if ind_x.size > 0:
                    i = i + 1
                    continue

                if ta > tz and (xa != x or xa != y):
                    pax = xa
                    pay = ya
                    hist = np.vstack([hist, np.array([pay,pax])])
                    da = 1
                    xaa = xa
                    yaa = ya

                i = i + 1

            if da == 0:
                break

        if pbo != 0 and not stop:

            cn = 0

            for ii in range(1,9):
                [t1, x_A, y_A] = p(img, pbx,pby,ii)
                [t2, x_B, y_B] = p(img, pbx, pby, ii + 1)
                cn = cn + abs(t1 - t2)

            cn = cn/2.0

            if cn == 1 or cn == 3:
                stop = True

            i = 1

            while i < 9 and db == 0:
                [ta,xa,ya] = p(img, pbx, pby, i)
                [tz,xz,yz] = p(img, pbx, pby, i + 1)

                ind_y = np.where(hist[:,0] == ya)[0]
                ind_x = np.where(hist[ind_y,1] == xa)[0]
                if ind_x.size > 0:
                    i = i + 1
                    continue

                if ta > tz and (xa != x or xa != y):
                    pbx = xa
                    pby = ya
                    hist = np.vstack([hist,[pby,pbx]])
                    db = 1
                    xbb = xa
                    ybb = ya

                i = i + 1

                
        if pco != 0 and not stop:
            cn = 0
            for ii in range(1,9):
                [ta, x_A, y_A] = p(img, pcx, pcy, ii)
                [tz, x_B, y_B] = p(img, pcx, pcy, ii+1)
                cn = cn + abs(t1 - t2)
            cn = cn/2.0

            if cn == 1 or cn == 3:
                stop = True

            i = 1
            while i < 9 and dc == 0:

                [ta, xa, ya] = p(img, pcx, pcy, i)
                [tz, xz, yz] = p(img, pcx, pcy, i + 1)

                ind_y = np.where(hist[:,0] == ya)[0]
                ind_x = np.where(hist[ind_y,1] == xa)[0]

                if ind_x.size > 0:
                    i = i + 1
                    continue

                if ta > tz and (xa != x or xa != y):
                    pcx = xa
                    pcy = ya
                    hist = np.vstack([hist, np.array([pcy,pcx])])
                    dc = 1
                    xcc = xa
                    ycc = ya

                i = i + 1

    t1 = np.tile(np.array([xaa,yaa]),[1,1]).transpose()
    t2 = np.tile(np.array([xbb,ybb]),[1,1]).transpose()
    t3 = np.tile(np.array([xcc,ycc]),[1,1]).transpose()

    d1 = np.sqrt(dist2(t1,t2))
    d2 = np.sqrt(dist2(t1,t3))
    d3 = np.sqrt(dist2(t3,t2))

    if d1 >= d3 and d2 >= d3:
        sx = xaa
        sy = yaa
        ind = pao
    elif d1 >= d2 and d3 >= d2:
        sx = xbb
        sy= ybb
        ind = pbo
    elif d3 >= d1 and d2 >= d1:
        sx = xcc
        sy = ycc
        ind = pco
    else:
        time.pause()


    t1 = np.tile(np.array([xaa,yaa]),[1,1]).transpose()
    l1 = np.tile(np.array([xbb,ybb]),[1,1]).transpose()
    r1 = np.tile(np.array([xcc,ycc]),[1,1]).transpose()
    
    t2 = np.tile(np.array([core_x,core_y]),[1,1]).transpose()

    d1 = np.sqrt(dist2(t1,t2))
    d2 = np.sqrt(dist2(l1,t2))
    d3 = np.sqrt(dist2(r1,t2))

    qx = 0
    qy = 0
    diff = 0

    if d1 >= d2 and d1 >= d3:
        qx = xaa
        qy = yaa
        ind = pao
    elif d2 >= d3 and d2 >= d1:
        qx = xbb
        qy = ybb
        ind = pbo
    elif d3 >= d2 and d3 >= d1:
        qx = xcc
        qy = ycc
        ind = pco
    else:
        time.pause

    angle = np.mod(math.atan2(y - sy, sx - x),2*math.pi)

    res = 3

    return res, progress, sx, sy, angle
    
   
            

    
