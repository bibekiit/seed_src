from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import scipy.io
from scipy.sparse import bsr_matrix
import math, time
import dist2 as ds
import sc_compute as sc
import hist_cost_2 as hc
import hungarian2 as h2
import matplotlib.pyplot as plt
import bookstein as bs
import pdb

#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#    Description: Performs a heavily modified iterative TPS warping based the freely available
#    Shape Context code by (paper found at http://www.eecs.berkeley.edu/Research/Projects/CS/vision/shape/pami.html)

def tps_iter_match_1(m1, m2, X, Y, orients, nbins_theta, nbins_r, r_inner, r_outer, n_iter, r, beta_init, i1, i2, anchors):

    """script for doing shape-context based matching with alternating steps
     of estimating correspondences and estimating the regularized TPS
     transformation
    anchors=[];
    """

    gcx = i1
    gcy = i2
    l_a = 0
    angle = 0
    angle_b = 0
    aff_cost = 0
    MAX_ANGLE = 30

    if anchors.size > 0:
        gcx = np.hstack([np.tile(anchors[:, 0].T,(1,1)),i1])
        gcy = np.hstack([np.tile(anchors[:, 1].T,(1,1)),i2])
        l_a = max(anchors[:, 0].T.shape)
        mm1 = m1[(anchors[:, 0].T -1)]
        mm2 = m2[(anchors[:, 1].T -1)]
        x = X[(anchors[:, 0].T -1)]
        y = Y[(anchors[:, 1].T -1)]
    else:
        mm1 = np.array([])
        mm2 = np.array([])
        x = np.array([])
        y = np.array([])

    m1 = m1[(gcx -1), :][0,:]
    m2 = m2[(gcy -1), :][0,:]

    ndum1 = 0
    ndum2 = 0
    eps_dum = 1
    nsamp1 = gcx.size
    nsamp2 = gcy.size
    X = X[(gcx -1), :][0,:]
    Y = Y[(gcy -1), :][0,:]

    mean_dist_global = np.array([])
    display_flag = 0
    Et = 0
    global OLD_METHOD
    OLD_METHOD = 0

    if nsamp2 > nsamp1:
        # (as is the case in the outlier test)                                    
        ndum1 = ndum1 + (nsamp2 - nsamp1)
    eps_dum = 1

    #JI: store minimum error in iterations to avoid 
    # large jumps away from global minima to local minima
    min_error = - 1

    # initialize transformed version of model pointset
    Xk = X.copy()

    # initialize counter
    k = 1
    s = 1
    # out_vec_{1,2} are indicator vectors for keeping track of estimated
    # outliers on each iteration
    out_vec_1 = np.zeros(shape=(1, nsamp1), dtype='float16')
    out_vec_2 = np.zeros(shape=(1, nsamp2), dtype='float16')
    cvec = np.array([])
    
    while s:

        #disp(['iter=' int2str(k)])
        # compute shape contexts for (transformed) model
        #   [BH1,mean_dist_1]=sc_compute(Xk', m1(:,4)' ,mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1);

        BH1, mean_dist_1 = sc.sc_compute(Xk.T, np.zeros(shape=(1, nsamp1), dtype='float16'), mean_dist_global, nbins_theta, nbins_r, r_inner, r_outer, out_vec_1) # nargout=2
        
        # compute shape contexts for target, using the scale estimate from
        # the warped model
        # Note: this is necessary only because out_vec_2 can change on each
        # iteration, which affects the shape contexts.  Otherwise, Y does
        # not change.

        #   [BH2,mean_dist_2]=sc_compute(Y',m2(:,4)',mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2);
        BH2, mean_dist_2 = sc.sc_compute(Y.T, np.zeros(shape=(1, nsamp2), dtype='float16'), mean_dist_1, nbins_theta, nbins_r, r_inner, r_outer, out_vec_2) # nargout=2

        # compute regularization parameter
        beta_k = np.dot(np.dot((mean_dist_1 ** 2), beta_init), r ** (k - 1))

        # compute pairwise cost between all shape contexts
        costmat = hc.hist_cost_2(BH1, BH2, r_inner, r_outer, nbins_theta, nbins_r)

        # ensure that no negative entries in the cost matrix
        eps = np.spacing(1)
        temp = costmat < eps
        costmat[temp.nonzero()]=eps

        # pad the cost matrix with costs for dummies
        nptsd = nsamp1 + ndum1
        costmat2 = np.dot(eps_dum, np.ones(shape=(nptsd, nptsd), dtype='float16'))

        #JI: cost matrix with dummy nodes appended
        costmat2[0:nsamp1, 0:nsamp2] = costmat
        #disp('running hungarian alg.')

        costmat2 = costmat2[(l_a + 1 -1):nsamp2, (l_a + 1 -1):nsamp2]

        dist_m = ds.dist2(X[(l_a + 1 -1):nsamp1], Y[(l_a + 1 -1):nsamp2])
        for i in range(1, (nsamp1 - l_a +1)):
            for j in range(1, (nsamp2 - l_a +1)):
                intersect_flag = 0
                for li in range(1, (l_a +1)):
                    x1 = X[(li -1), 0]
                    x2 = Y[(li -1), 0]
                    x3 = X[(i -1), 0]
                    x4 = Y[(j -1), 0]
                    y1 = X[(li -1), 1]
                    y2 = Y[(li -1), 1]
                    y3 = X[(i -1), 1]
                    y4 = Y[(j -1), 1]

                    if np.dot(np.linalg.det(np.array([[1, 1, 1], [x1, x2, x3], [y1, y2, y3]])), np.linalg.det(np.array([[1, 1, 1],[x1, x2, x4], [y1, y2, y4]]))) <= 0 and np.dot(np.linalg.det(np.array([[1, 1, 1], [x1, x3, x4], [y1, y3, y4]])), np.linalg.det(np.array([[1, 1, 1], [x2, x3, x4], [y2, y3, y4]]))) <= 0:
                        intersect_flag = 1
                if intersect_flag and OLD_METHOD == 1:
                    costmat2[(i -1), (j -1)] = costmat2[(i -1), (j -1)] + 1000
                    continue

                mox = m1[(i -1), 3]
                moy = m2[(j -1), 3]
                z = np.minimum(abs(mox - moy), abs(np.dot(2, np.pi) - abs(mox - moy)))
                #	   costmat2(i,j)
                #	   dist_m(i,j)

                if OLD_METHOD == 0:
                    costmat2[(i -1), (j -1)] = (costmat2[(i -1), (j -1)] + 1 / np.exp(np.dot(- 5, dist_m[(i -1), (j -1)])) + 1 / np.exp(- z)) / 5
                    #-max(orients(i,j),0.2);
                else:
                    costmat2[(i -1), (j -1)] = np.dot(0.1, costmat2[(i -1), (j -1)]) + dist_m[(i -1), (j -1)]
                    #-max(orients(i,j),0.2);

                if z < np.pi / 8 and dist_m[(i -1), (j -1)] < 0.06:
                    costmat2[(i -1), (j -1)] = costmat2[(i -1), (j -1)] / 2
                    #                 else
                    #                    costmat2(i,j)=costmat2(i,j) + 5*abs(costmat2(i,j));
        ##                if m1[(i -1), 3] == m2[(j -1), 3]:
        ##                    #             costmat2(i,j)=costmat2(i,j)*0.95;
        
        if costmat2.size > 1:
            [cvec,Tcost] = h2.hungarian(costmat2)
            cvec = cvec + l_a

            for tt in range(1, (cvec.size +1)):
                if costmat2[(cvec[(tt -1)] - l_a - 1 ), (tt -1)] > 1.4:
                    #print costmat2[(cvec[(tt -1)] - l_a - 1 ), (tt -1)]
                    cvec[(tt -1)] = nsamp2
        else:
            orient_res = - 1
            mse2 = 1
            sc_cost = 1
            E = 0
            theta_offset_by_warping = 50
            return sc_cost, Et, cvec, angle

        cvec = np.hstack([np.array(range(1, l_a+1)),cvec])

        # update outlier indicator vectors
        a = np.sort(cvec) # nargout=2
        cvec2 = np.argsort(cvec)
        # out_vec_1=cvec2(1:nsamp1)>nsamp2;
        # out_vec_2=cvec(1:nsamp2)>nsamp1;             
        #JI: cvec points to X match
        # format versions of Xk and Y that can be plotted with outliers'
        # correspondences missing

        X2 = np.dot(np.nan, np.ones(shape=(nptsd, 2), dtype='float16'))
        m1a = np.dot(np.nan, np.ones(shape=(nsamp2, 6), dtype='float16'))
        X2[0:nsamp1, :] = Xk
        m1a[0:nsamp1, :] = m1
        X2 = X2[(cvec -1), :]

        m1a = m1a[(cvec -1), :]
        X2b = np.dot(np.nan, np.ones(shape=(nptsd, 2), dtype='float16'))
        X2b[0:nsamp1, :] = X
        X2b = X2b[(cvec -1), :]
        m2a = m2

        Y2 = np.dot(np.nan, np.ones(shape=(nptsd, 2), dtype='float16'))
        Y2[0:nsamp2, :] = Y

        # extract coordinates of non-dummy correspondences and use them
        # to estimate transformation
        ind_good = np.intersect1d(np.flatnonzero(np.invert(np.isnan(X2b[:, 0]))), np.flatnonzero(np.invert(np.isnan(Y2[:, 0]))))
        #   ind_good=intersect(find(~isnan(X2b(:,1))), find(~isnan(Y2(:,1))))

        dd = ds.dist2(X2b[(ind_good), :], Y2[(ind_good), :])
        i_vgood = range(1, (l_a +1))
        #[];
        for i in range(l_a + 1, (np.minimum(ind_good.size, nsamp1) +1)):
            if dd[(i -1), (i -1)] < 0.1:
                i_vgood = np.concatenate([i_vgood, np.array([i])])
        ind_good = ind_good[(i_vgood -1)] + 1

        n_good = max(ind_good.shape)

        X3b = X2b[(ind_good -1), :]
        Y3 = Y2[(ind_good -1), :]

        m1a = m1a[(ind_good -1), :]
        m2a = m2a[(ind_good -1), :]

        n_good = max(ind_good.shape)


        if display_flag == 0:
            plt.subplot(2, 2, 2)
            plt.cla()
            plt.plot(X2[:, 0], X2[:, 1], 'b+', Y2[:, 0], Y2[:, 1], 'ro')
            plt.hold('on')
            h = plt.plot(np.concatenate((np.tile(X2[:, 0],(1,1)),np.tile(Y2[:, 0],(1,1)))), np.concatenate((np.tile(X2[:, 1],(1,1)),np.tile(Y2[:, 1],(1,1)))), 'k-')
            plt.hold('off')
            plt.title(str(n_good) + ' correspondences (warped X)')
            plt.draw()
            time.sleep(2)
            
            
            # show the correspondences between the untransformed images
            plt.subplot(2, 2, 3)
            plt.cla()
            plt.plot(X[:, 0], X[:, 1], 'b+', Y[:, 0], Y[:, 1], 'ro')
            plt.ind = cvec[(ind_good -1)]
            plt.hold('on')
            plt.plot(np.concatenate((np.tile(X2b[:, 0],(1,1)),np.tile(Y2[:, 0],(1,1)))), np.concatenate((np.tile(X2b[:, 1],(1,1)),np.tile(Y2[:, 1],(1,1)))), 'k-')
            plt.hold('off')
            plt.title(str(n_good) + ' correspondences (unwarped X)')
            plt.draw()
            time.sleep(2)
            

        # estimate regularized TPS transformation
        if X3b.size > 1:
            cx, cy, E = bs.bookstein(X3b, Y3, beta_k) # nargout=3
            if np.isnan(cx[0]):
                mse2 = 1
                sc_cost = 1
                break
        else:
            mse2 = 1
            sc_cost = 1
            E = 50
            return sc_cost, Et, cvec, angle
        Et = Et + E

        # calculate affine cost
        A = np.concatenate((np.tile(cx[n_good+1:n_good+3],(1,1)), np.tile(cy[n_good + 2 -1:n_good + 3],(1,1)))).T
        print 'A = ' + str(A) + '\n'
        print str(np.tile(cx[n_good+1:n_good+3],(1,1))) + '\n'
        print str(A[1,0]) + '\n'
        print str(A[0,0]) + '\n'
        
        U, S, V = np.linalg.svd(A) # nargout=3

        angle_b = angle_b + ((np.dot(math.atan2(V[0, 1], V[0, 0]), 180) / np.pi) - (np.dot(math.atan2(U[0, 1], U[0, 0]), 180) / np.pi))
        print str((np.dot(math.atan2(V[0, 1], V[0, 0]), 180) / np.pi)) + '\n'
        print str((np.dot(math.atan2(U[0, 1], U[0, 0]), 180) / np.pi)) + '\n'

        # JA: Compute the eigenvalues of A in an array, ordered descending.
        Sdup = S
        print 'S = ' + str(Sdup) + '\n'
        aff_cost = aff_cost + math.log(Sdup[0] / Sdup[1])
        print 'aff_cost = ' + str(aff_cost) + '\n'
        #angle=abs(atan2( A(1,2) ,  A(2,2))*180/pi)
        angle = angle + (np.dot(math.atan2(A[1, 0], A[0, 0]), 180) / np.pi)

        orient_m = 0
        sc_cost = np.array([])
        index = 0
        m_index = 0
        bad_count = 0

        # warp each coordinate
        fx_aff = np.dot(np.tile(cx[(n_good + 1 -1):n_good + 3],(1,1)), np.concatenate([np.ones(shape=(1, nsamp1), dtype='float16'), X.T]))
        d2 = np.maximum(ds.dist2(X3b, X), 0)
        U = d2 * np.log(d2 + np.spacing(1))
        fx_wrp = np.dot(cx[0:n_good].T, U)
        fx = fx_aff + fx_wrp
        fy_aff = np.dot(np.tile(cy[(n_good + 1 -1):n_good + 3],(1,1)), np.concatenate([np.ones(shape=(1, nsamp1), dtype='float16'), X.T]))
        fy_wrp = np.dot(np.tile(cy[0:n_good],(1,1)), U)
        fy = fy_aff + fy_wrp

        Z = np.concatenate((fx, fy)).T

        Zk = np.dot(np.nan, np.ones(shape=(nptsd, 2), dtype='float16'))
        Zk[0:nsamp1] = Z
        Zk = Zk[(cvec -1)]
        Zk = Zk[(ind_good -1)]

        # compute theta_offset_by_warping from Xk and Z, and update
        # theta_offset_total
        Diff = np.subtract(Xk,Z)
        diff_x = np.sum(Diff[:, 0])
        diff_y = np.sum(Diff[:, 1])

        # compute the mean squared error between synthetic warped image
        # and estimated warped image (using ground-truth correspondences
        # on TPS transformed image) 
        #  mse2=sqrt(mean((Y3(:,1)-Z(:,1)).^2+(Y3(:,2)-Z(:,2)).^2) );
        # mse2=sqrt(mean((Y3(:,1)-Zk(:,1)).^2+(Y3(:,2)-Zk(:,2)).^2) );
        mse2 = 0
        #   disp(['error = ' num2str(mse2)])

        if 0:
            #display_flag == 0
            plt.plot(Z[:, 0], Z[:, 1], 'b+', Y[:, 0], Y[:, 1], 'ro')
            plt.title('recovered TPS transformation (k=' + str(k) + ', \\lambda_o=' + str(np.dot(beta_init, r ** (k - 1))) + ', I_f=' + str(E) + ', error=' + str(mse2) + ')')
            # show warped coordinate grid
            plt.hold('on')
            plt.plot(fx, fy, 'k.', 'markersize', 1)
            plt.hold('off')
            plt.draw()

        # update Xk for the next iteration
        Xk = Z

        # stop early if shape context score is sufficiently low
        if k == n_iter or Et > 1000:
            s = 0
        else:
            k = k + 1

    #Y(cvec(l_a+1:nsamp1),:)
    #Xk(cvec(l_a+1:nsamp1),:)
    #X(cvec(l_a+1:nsamp1),:)

    cvec = cvec[(l_a + 1 -1):nsamp2]

    print 'angle = ' + str(angle) + '\n'
    angle_b = np.minimum(abs(angle_b), 360 - abs(angle_b))
    print 'angle_b = ' + str(angle_b) + '\n'

    #pause
    map_ = np.array([])
    index = 1
    angle = angle_b
    print 'aff_cost = ' + str(aff_cost) + '\n'
    
    #aff_cost=0.3 works well
    # [ig,ib]=calc_EER(1./(RES_G-0.3*SC_G+0.3),  1./(RES_B-0.3*SC_B+0.3)); hold on ; plot(ig,'b'); plot(ib,'r');hold off;   axis([0 75 0.5 1]);
    
    if np.logical_and(abs(angle) < MAX_ANGLE, np.logical_and(E < 15, aff_cost < 0.3)):

        #if angle < 30 && E < 15 || isnan(E)
        #if angle < 15 && Et < 13 || isnan(E)
        for i in range(1, (nsamp2 - l_a +1)):
            if cvec[(i -1)] <= nsamp1:
                intersect_flag = 0

                d1 = math.sqrt((X[(cvec[(i -1)] -1), 0] - Y[(i -1), 0]) ** 2 + (X[(cvec[(i -1)] -1), 1] - Y[(i -1), 1]) ** 2)
                d1 = math.sqrt((Xk[(cvec[(i -1)] -1), 0] - Y[(i -1), 0]) ** 2 + (X[(cvec[(i -1)] -1), 1] - Y[(i -1), 1]) ** 2)
                dist_m = ds.dist2(X[(l_a + 1 -1):nsamp1], Y[(l_a + 1 -1):nsamp2])

                #           if OLD_METHOD==1	
                for li in range(1, (l_a +1)):
                    x1 = X[(li -1), 0]
                    x2 = Y[(li -1), 0]
                    x3 = X[(cvec[(i -1)] -1), 0]
                    x4 = Y[(i -1), 0]
                    y1 = X[(li -1), 1]
                    y2 = Y[(li -1), 1]
                    y3 = X[(cvec[(i -1)] -1), 1]
                    y4 = Y[(i -1), 1]

                    if np.dot(np.linalg.det(np.array([[1, 1, 1], [x1, x2, x3], [y1, y2, y3]])), np.linalg.det(np.array([[1, 1, 1], [x1, x2, x4], [y1, y2, y4]]))) <= 0 and np.dot(np.linalg.det(np.array([[1, 1, 1], [x1, x3, x4], [y1, y3, y4]])), np.linalg.det(np.array([[1, 1, 1], [x2, x3, x4], [y2, y3, y4]]))) <= 0:
                        intersect_flag = 1
                        print 'intersect_flag' + str(intersect_flag) + '\n'

                if intersect_flag == 1 and costmat[(cvec[(i -1)] -1), (i -1)] > 1:
                    ll = costmat[(cvec[(i -1)] -1), (i -1)]
                    print 'l1 = ' + str(l1) + '\n'
                    continue

                if (costmat[(cvec[(i -1)] -1), (i -1)] < 1.31 and OLD_METHOD == 0) or (d1 < 0.013 and intersect_flag == 0 and OLD_METHOD == 1):
                    # && costmat(cvec(i), i) < 20
                    map_[(index -1)] = np.array([cvec[(i -1)] - l_a, i])
                    index = index + 1
                else:
                    ll = costmat[(cvec[(i -1)] -1), (i -1)]
                    print 'l1 = ' + str(l1) + '\n'

    cvec = map_

    return [sc_cost, Et, cvec, angle]

##m1 = loadmat('m1.mat')['m1']
##m2 = loadmat('m2.mat')['m2']
##X = loadmat('X.mat')['X']
##Y = loadmat('Y.mat')['Y']
##orients = loadmat('orients.mat')['orients']
##nbins_theta = 19
##nbins_r = 5
##r_inner = 0.1250
##r_outer = 0.5000
##n_iter = 3
##r = 0.3500
##beta_init = 0.8000
##i1 = loadmat('i1.mat')['i1']
##i2 = loadmat('i2.mat')['i2']
##anchors = loadmat('anchors.mat')['anchors']
##[sc_cost, Et, cvec, angle] = tps_iter_match_1(m1, m2, X, Y, orients, nbins_theta, nbins_r, r_inner, r_outer, n_iter, r, beta_init, i1, i2, anchors)

