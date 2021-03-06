from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os
import pdb
import math, threading, time
import register as rs
import dist2 as ds
from calc_orient import calc_orient
from sc_compute import sc_compute
from hist_cost_2 import hist_cost_2
import tps_iter_match as tim
import matplotlib.pyplot as plt
#############################################################################
# Author - Bibek Behera                                                     #
# Date - Feb, 2014                                                          #
# Place - IIT Bombay                                                        #
#############################################################################
#    Description: match a test fingerprint of image 'f1' to a template fingerprint of image 'f2'.
#                 Preprocessing and feature extraction must already be performed (see extract_db.m).
#
#
def do_match(*varargin):
    
    nargin = len(varargin)
    if nargin > 0:
        f1 = varargin[0]
    if nargin > 1:
        f2 = varargin[1]
    if nargin == 0:
        f1 = 'DB1_B\\105_4.tif'
        f2 = 'DB1_B\\105_6.tif'
    #load template fingerprint
    X = np.genfromtxt(f1+'.X',delimiter=',')
    m1 = np.genfromtxt(f1+'.m',delimiter=',')
    oX = np.genfromtxt(f1+'.o',delimiter=',')
    rX = np.genfromtxt(f1+'.r',delimiter=',')
    nX = np.genfromtxt(f1+'.n',delimiter=',')
    roX = np.genfromtxt(f1+'.ro',delimiter=',')
    orient_img_x = np.array([]) #load( [char(f1) '.oi']);
    or_x = np.array([]) #load( [char(f1) '.oi']);

    #load test fingerprint
    Y = np.genfromtxt(f2+'.X',delimiter=',')
    m2 = np.genfromtxt(f2+'.m',delimiter=',')
    oY = np.genfromtxt(f2+'.o',delimiter=',')
    rY = np.genfromtxt(f2+'.r',delimiter=',')
    nY = np.genfromtxt(f2+'.n',delimiter=',')
    roY = np.genfromtxt(f2+'.ro',delimiter=',')
    orient_img_y = np.array([])     #load( [char(f2) '.oi']);
    or_y = np.array([])    #load( [char(f2) '.oi']);

    
    #Make sure all minutiae type are ridge endings, bifurcations, or primary cores (pseudo minutia).
    i1 = np.flatnonzero(m1[:, 2] < 7) + 1
    i2 = np.flatnonzero(m2[:, 2] < 7) + 1
    X = X[(i1 -1), :]
    m1 = m1[(i1 -1), :]
    Y = Y[(i2 -1), :]
    m2 = m2[(i2 -1), :]
    m1r = m1[:, 3].copy()
    m2r = m2[:, 3].copy()
    m1[:, 3] = np.remainder(m1[:, 3], np.pi)
    m2[:, 3] = np.remainder(m2[:, 3], np.pi)


    #Attempt to retrieve index of both test and template fingerprint core indexes (provided they exist).
    test = m1[:, 2].copy()
    ic1 = np.flatnonzero(test == 5) + 1
    c1o = - 1
    c2o = - 1
    test = m2[:, 2].copy()
    ic2 = np.flatnonzero(test == 5) + 1


    #Build fingerprint feature structure.
    f1 = {'X': X, 'M': m1, 'O': oX, 'R': rX, 'N': nX, 'RO':roX, 'OIMG': orient_img_x, 'OREL': or_x}
    f2 = {'X': Y, 'M': m2, 'O': oY, 'R': rY, 'N': nY, 'RO':roY, 'OIMG': orient_img_y, 'OREL': or_y}

    #Bending energy
    E = 0

    #Detect if the core is near the edge of the fingerprint. Edge core comparisons have their similarity 
    #reduced since they are more prone to error due to lack of distributed coverage about the core point 
    #(i.e. much more possible alignment combinations are achieved when minutiae only lie on one side 
    # of a core point since we have less rotational contraints).
    edge_core = 0
    if ic1.size > 0 and ic2.size > 0:
        a = f1['X'][:,0].max()
        b = f1['X'][:,0].min()
        if f1['X'][ic1-1, 0] > a - 0.02 or f1['X'][ic1-1, 0] < b + 0.02:
            edge_core = 1
        a = f2['X'][:, 0].max()
        b = f2['X'][:, 0].min()
        if f2['X'][ic2-1, 0] > a - 0.02 or f2['X'][ic2-1, 0] < b + 0.02:
            edge_core = 1

    #Swap structures to make sure X represents the fingerprint with fewer minutiae
    if X.shape[0] > Y.shape[0]:
        temp = f1
        f1 = f2
        f2 = temp
        temp = ic1
        ic1 = ic2
        ic2 = temp
        temp = m1r
        m1r = m2r
        m2r = temp

    display_flag = 0
    GC = 0

    mean_dist_global = np.array([])    # use [] to estimate scale from the data
    
    nbins_theta = 19
    nbins_r = 5
    eps_dum = 1
    r = 0.35     # annealing rate
    beta_init = 0.8     # initial regularization parameter (normalized)
    
    r_inner = 1 / 8
    r_outer = 1 / 2

    #Register fingerprints
    [ft, res] = rs.register(f1, f2) # nargout=2
    angle = 0
    nsamp1 = f1['X'].shape[0]
    nsamp2 = f2['X'].shape[0]
    out_vec_1 = np.zeros(shape=(1, nsamp1), dtype='float16')
    out_vec_2 = np.zeros(shape=(1, nsamp2), dtype='float16')

    d1 = ds.dist2(f1['X'], f1['X'])
    d2 = ds.dist2(f2['X'], f2['X'])

    o_res = np.array([])
    sc_res = np.array([])
    mc_res = np.array([])
    ro_res = np.array([])

    #Map out binding box for overlap region.
    xt = np.min(ft['X'][:, 1]) 
    xb = np.max(ft['X'][:, 1])
    xl = np.min(ft['X'][:, 0])
    xr = np.max(ft['X'][:, 0])
    yt = np.min(f2['X'][:, 1])
    yb = np.max(f2['X'][:, 1])
    yl = np.min(f2['X'][:, 0])
    yr = np.max(f2['X'][:, 0])
    region_t = max(xt, yt)
    region_b = min(xb, yb)
    region_r = min(xr, yr)
    region_l = max(xl, yl)
    region_a = np.dot((region_b - region_t), (region_r - region_l))

    #Find all indices within bounding box
    ind1 = np.intersect1d(np.intersect1d(np.flatnonzero(ft['X'][:, 0] > region_l), np.flatnonzero(ft['X'][:, 0] < region_r)), np.intersect1d(np.flatnonzero(ft['X'][:, 1] > region_t), np.flatnonzero(ft['X'][:, 1] < region_b)))
    ind2 = np.intersect1d(np.intersect1d(np.flatnonzero(f2['X'][:, 0] > region_l), np.flatnonzero(f2['X'][:, 0] < region_r)), np.intersect1d(np.flatnonzero(f2['X'][:, 1] > region_t), np.flatnonzero(f2['X'][:, 1] < region_b)))

    #get minutiae count for each image in overlap region
    ng_samp1 = ind1.size
    ng_samp2 = ind2.size

    if res['map'].size > 0:
        if ic1.size > 0 and ic2.size > 0:
            GC = 1

        f1 = ft.copy()
        inda1 = res['map1'].copy()
        inda2 = res['map2'].copy()

        #find tighter minutiae count if possible for non core images 
        #which are more likely to have a much smaller overlap area 
        #then suggested by the bounding box. Convex hull structures
        #may be more accurate here.

        if GC != 1 and np.dot(ng_samp1, ng_samp2) > np.dot(inda1.size, inda2.size):
            # overlap region minutiae counts are set to nearest neighbour minutiae index counts
            ng_samp1 = inda1.size
            ng_samp2 = inda2.size

        # overlap region index sets have anchor minutiae indexes removed.
        inda1 = np.setdiff1d(inda1[np.flatnonzero(f1['M'][inda1-1, 2] < 5)], res['map'][:, 0])
        inda2 = np.setdiff1d(inda2[np.flatnonzero(f2['M'][inda2-1, 2] < 5)], res['map'][:, 1])

        y = 0
        redo = np.array([])
        #   for i=1:size(res['map'],1)
        #       a1=mod(m1r[res['map'][i -1,0])-res.angle,2*pi);
        #       a2=m2r[res['map'][i -1,2));
        #       if min(abs(a1-a2), 2*pi-abs(a1-a2)) > pi/8 && f1['M'][res['map'][i -1,0],2] == f2['M'][res['map'][i -1,1],2]
        #          y=y+1;
        #          redo(y)=i;
        #          inda1(numel(inda1)+1)=res['map'][i -1,1);
        #          inda2(numel(inda2)+1)=res['map'][i -1,1];
        #       end
        #   end

        res['map'] = res['map'][np.setdiff1d(range(1, (res['map'].shape[0] +1)), redo)-1, :]
        f1['M'][:, 3] = np.mod(m1r - res['angle'], np.dot(2, np.pi))

        orients = np.zeros((inda1.size, inda2.size))
        for i in range(1, (inda1.size +1)):
            for j in range(1, (inda2.size +1)):
                if f1['M'][inda1[(i -1)]-1, 2] < 5 and f2['M'][inda2[(j -1)]-1, 2] < 5:
                    orients[(i -1), (j -1)] = calc_orient(np.tile(f1['X'][inda1[(i -1)]-1, :],(1,1)), np.tile(f1['R'][inda1[(i -1)]-1, :],(1,1)), np.tile(f2['X'][inda2[(j -1)]-1, :],(1,1)), np.tile(f2['R'][inda2[(j -1)]-1, :],(1,1)))[0]
                else:
                    orients[(i -1), (j -1)] = 0
        
        if np.logical_and(inda1.size > 1, inda2.size > 1):
            if inda1.size > inda2.size:
                orients = orients.T
                t_res_map = res['map'][:,[1, 0]]
                [sc_cost3, E, cvec, angle] = tim.tps_iter_match_1(f2['M'], f1['M'], f2['X'], f1['X'], orients, nbins_theta, nbins_r, r_inner, r_outer, 3, r, beta_init, np.tile(inda2,(1,1)), np.tile(inda1,(1,1)), t_res_map) # nargout=4
                if cvec.size > 0:
                    cvec = np.tile(cvec,(1,1))
                    xx = np.vstack((inda1[(cvec[:, 1] -1)], inda2[(cvec[:, 0] -1)])).T
                    res['map'] = np.vstack((res['map'], xx))
            else:
                [sc_cost3, E, cvec, angle] = tim.tps_iter_match_1(f1['M'], f2['M'], f1['X'], f2['X'], orients, nbins_theta, nbins_r, r_inner, r_outer, 3, r, beta_init, np.tile(inda1,(1,1)), np.tile(inda2,(1,1)), res['map']) # nargout=4
                if cvec.size > 0:
                    cvec = np.tile(cvec,(1,1))
                    xx = np.vstack((inda1[(cvec[:, 0] -1)], inda2[(cvec[:, 1] -1)])).T
                    res['map'] = np.vstack((res['map'], xx))

        d1 = np.sqrt(ds.dist2(f1['X'], f1['X']))
        d2 = np.sqrt(ds.dist2(f2['X'], f2['X']))

        s1 = np.flatnonzero(f1['M'][:, 2] < 5).size
        s2 = np.flatnonzero(f2['M'][:, 2] < 5).size
    else:
        res['map'] = np.array([])
        res['o_res'] = np.array([])

    nX = np.array([])
    nY = np.array([])

    ns = 3
    #n_weight = np.array([])
    n_weight = []
    #same_type = np.array([])
    same_type = []

    if res['map'].size > 0:
        for i in range(1, (res['map'].shape[0] +1)):
            if res['map'][i -1, 0] == 0:
                continue
            x = m1[(res['map'][i -1, 0] -1), 0]
            y = m1[(res['map'][i -1, 0] -1), 1]

            bonus = 1
            a1 = np.mod(m1r[(res['map'][i -1, 0] -1)] - res['angle'], np.dot(2, np.pi))
            a2 = m2r[(res['map'][i -1, 1] -1)]

            bonus = np.dot(bonus, math.exp(- np.minimum(abs(a1 - a2), np.dot(2, np.pi) - abs(a1 - a2))))

            if np.logical_and(GC == 1, f1['M'][res['map'][i -1, 0]-1, 2] != f2['M'][res['map'][i -1, 1]-1, 2]):
                bonus = bonus - 0.1
                same_type.append(0)
            else:
                same_type.append(1)

            dx = np.sort(d1[(res['map'][i -1, 0] -1), :]) # nargout=2
            ii = np.argsort(d1[(res['map'][i -1, 0] -1), :])
            dy = np.sort(d2[(res['map'][i -1, 1] -1), :]) # nargout=2
            jj = np.argsort(d2[(res['map'][i -1, 1] -1), :])
            dd1 = f1['N'][np.ix_(np.array(range(np.dot((res['map'][i -1, 0] - 1), (s1 - 1)) + 1, (np.dot((res['map'][i -1, 0] - 1), (s1 - 1)) + ns +1)))-1, np.array(range(3, 10))-1)]
            dd2 = f2['N'][np.ix_(np.array(range(np.dot((res['map'][i -1, 1] - 1), (s2 - 1)) + 1, (np.dot((res['map'][i -1, 1] - 1), (s2 - 1)) + ns +1)))-1, np.array(range(3, 10))-1)]
            dd1[:, 2] = np.mod(dd1[:, 2] - res['angle'], np.dot(2, np.pi))
            dd2[:, 2] = np.mod(dd2[:, 2], np.dot(2, np.pi))
            dd1[:, 6] = np.mod(dd1[:, 2], np.pi)
            dd2[:, 6] = np.mod(dd2[:, 2], np.pi)

            used = []
            m_score = 0
            t = 1
            for x in range(1, (ns +1)):
                for y in range(1, (ns +1)):
                    if np.flatnonzero(np.array(used) == y).size:
                        continue
                    a_diff = np.minimum(abs(dd1[(x -1), 2] - dd2[(y -1), 2]), np.dot(2, np.pi) - abs(dd1[(x -1), 2] - dd2[(y -1), 2]))
                    dist_diff = abs(dd1[(x -1), 0] - dd2[(y -1), 0])
                    lo_diff = abs(dd1[(x -1), 5] - dd2[(y -1), 5])
                    o_diff = np.minimum(abs(dd1[(x -1), 6] - dd2[(y -1), 6]), np.pi - abs(dd1[(x -1), 6] - dd2[(y -1), 6]))
                    if np.logical_and(dist_diff < 0.05, a_diff < np.pi / 2):
                        m_score = m_score + np.dot(np.dot(math.exp(- o_diff), math.exp(- dist_diff)), math.exp(- a_diff))
                        used.append(y)
                        t = t + 1
            n_weight = np.insert(n_weight, i-1, m_score + bonus)

            rox = f1['RO'][res['map'][i -1, 0]-1, :]
            roy = f2['RO'][res['map'][i -1, 1]-1, :]
            mox = f1['M'][res['map'][i -1, 0]-1, 3]
            moy = f2['M'][res['map'][i -1, 1]-1, 3]
            z = np.dot(np.minimum(abs(mox - moy), abs(np.pi - abs(mox - moy))), 2) / np.pi

            t1 = np.flatnonzero(rox < 0)
            t2 = np.flatnonzero(roy < 0)
            t = np.hstack([t1,t2,np.array([8])])
            if np.min(t).size > 0:
                rox = rox[0:np.min(t)]
                roy = roy[0:np.min(t)]
                ro_res = np.insert(ro_res, i-1, np.max(abs(rox - roy)))
                if np.logical_and(ro_res[(i -1)] < 0.1, np.min(t+1) > 4):
                    n_weight[(i -1)] = n_weight[(i -1)] + 0.2
            else:
                ro_res = np.insert(ro_res, i-1, 0)

            [o_res_sim, ih, anony] = calc_orient(np.tile(f1['O'][res['map'][i -1, 0]-1, :],(1,1)), np.tile(f1['R'][res['map'][i -1, 0]-1, :],(1,1)), np.tile(f2['O'][res['map'][i -1, 1]-1, :],(1,1)), np.tile(f2['R'][res['map'][i -1, 1]-1, :],(1,1))) # nargout=2
            o_res = np.insert(o_res, i-1, o_res_sim)
            mc_res = np.insert(mc_res, i-1, z)
            sc_res = np.insert(sc_res, i-1, 1)#costmat[res['map'][i -1,1),res['map'][i -1,2));
            res['map'][np.setdiff1d(np.array([i-1]),np.flatnonzero(res['map'][:, 0] == res['map'][i -1, 0])), 0] = 0
            res['map'][np.setdiff1d(np.array([i-1]),np.flatnonzero(res['map'][:, 1] == res['map'][i -1, 1])), 0] = 0

    sc_cost = 100

    if o_res.size > 1:
        inda1 = np.union1d(inda1, res['map'][:, 0])
        inda2 = np.union1d(inda2, res['map'][:, 1])

        A = f1['X'][inda1-1, :]
        B = f2['X'][inda2-1, :]
        out_vec_1 = np.zeros(shape=(1, A.shape[0]), dtype='float16')
        out_vec_2 = np.zeros(shape=(1, B.shape[0]), dtype='float16')

        BH1, mean_dist_1 = sc_compute(A.T, f1['M'][inda1-1, 3].T, mean_dist_global, nbins_theta, nbins_r, r_inner, r_outer, out_vec_1) # nargout=2
        BH2, mean_dist_2 = sc_compute(B.T, f2['M'][inda2-1, 3].T, mean_dist_1, nbins_theta, nbins_r, r_inner, r_outer, out_vec_2) # nargout=2
        #compute pairwise cost between all shape contexts
        costmat = hist_cost_2(BH1, BH2, r_inner, r_outer, nbins_theta, nbins_r)
        sc_vals = np.array([])

        for i in range(1, (int(res['map'].size / 2) +1)):
            if costmat[(np.flatnonzero(inda1 == res['map'][i -1, 0])), (np.flatnonzero(inda2 == res['map'][i -1, 1]))].size > 0:
                sc_vals = np.insert(sc_vals, (i -1), costmat[(np.flatnonzero(inda1 == res['map'][i -1, 0])), (np.flatnonzero(inda2 == res['map'][i -1, 1]))])
            else:
                sc_vals = np.insert(sc_vals, (i -1), 10)
        sc_cost = np.mean(sc_vals)
        #/exp(-(0.7-max(sqrt(res['area']),0))) 
    
    unknown_c = np.flatnonzero(o_res == - 1).size

    ind = np.intersect1d(np.flatnonzero(o_res > 0.25), np.flatnonzero(ro_res < 0.897))

    for i in range(1, (ind.size +1)):
        dd1 = f1['N'][np.ix_(np.array(range(np.dot((res['map'][ind[(i-1)], 0] -1), (s1 - 1)) + 1, (np.dot((res['map'][ind[(i-1)], 0] -1), (s1 - 1)) + ns +1)))-1, range(2, 9))]
        dd2 = f2['N'][np.ix_(np.array(range(np.dot((res['map'][ind[(i-1)], 1] -1), (s2 - 1)) + 1, (np.dot((res['map'][ind[(i-1)], 1] -1), (s2 - 1)) + ns +1)))-1, range(2, 9))]
    
    o_res = o_res[ind]
    ro_res = ro_res[ind]
    mc_res = mc_res[ind]
    sc_res = sc_res[ind]
    n_weight = n_weight[ind]
    same_type = np.array(same_type)[ind]
    #o_res=o_res.*n_weight;

##    res['sc'] / res['area']
##    res['sc']
##    ro_res     #=1/(sum(ro_res)/numel(ro_res))
##    o_res     #=sum(o_res)/numel(o_res)
##    n_weight
##    mc_res     #=1/sum(mc_res)/numel(mc_res)
##    sc_res

    nX = np.intersect1d(nX, ind1)
    nY = np.intersect1d(nY, ind2)
##    ind.size
##    res['map'].shape[0]

    ns1 = nX.size
    ns2 = nY.size

    ng_samp1 = ng_samp1 - unknown_c #=ns1;     #ng_samp1+(ns1/2);
    ng_samp2 = ng_samp2 - unknown_c     #=ns2;     #ng_samp2+(ns2/2);
    
    ic1 = np.flatnonzero(f1['M'][:, 2] == 5)
    ic2 = np.flatnonzero(f2['M'][:, 2] == 5)

    #a=abs(ns1-ng_samp1) + abs(ns2-ng_samp2)
    #4

    o_a = np.zeros(shape=(ind.size, ind.size), dtype='float16')
    o_b = np.zeros(shape=(ind.size, ind.size), dtype='float16')
    for i in range(1, (ind.size +1)):
        for j in range(i + 1, (ind.size +1)):
            [o_a[(i -1), (j -1)], ia, ra] = calc_orient(np.tile(f1['O'][res['map'][i -1, 0] -1, :],(1,1)), np.tile(f1['R'][res['map'][i -1, 0] -1, :],(1,1)), np.tile(f1['O'][res['map'][j-1, 0] -1, :],(1,1)), np.tile(f1['R'][res['map'][j-1, 0] -1, :],(1,1))) # nargout=3
            [o_b[(i -1), (j -1)], ib, rb] = calc_orient(np.tile(f2['O'][res['map'][i -1, 1] -1, :],(1,1)), np.tile(f2['R'][res['map'][i -1, 1] -1, :],(1,1)), np.tile(f2['O'][res['map'][j-1, 1] -1, :],(1,1)), np.tile(f2['R'][res['map'][j-1, 1] -1, :],(1,1))) # nargout=3
            o_a = np.minimum(o_a, 1)
            o_b = np.minimum(o_b, 1)

            if (np.intersect1d(ra, rb).size < 120):
                o_a[(i -1), (j -1)] = 0
                o_b[(i -1), (j -1)] = 0

            o_a[(j -1), (i -1)] = o_a[(i -1), (j -1)]
            o_b[(j -1), (i -1)] = o_b[(i -1), (j -1)]

    vv = 1
    if np.logical_and((mc_res.size >= 2), res['area'] > - 1):
        plt.hold('on')
        #         figure(1)
        plt.subplot(2, 2, 4)
        plt.plot(f1['X'][:, 0], f1['X'][:, 1], 'b+', f2['X'][:, 0], f2['X'][:, 1], 'ro')
##        if GC == 1:
            #          plot(f1['X'](ic1,1),f1['X'](ic1,2),'g+',f2['X'](ic2,1),f2['X'](ic2,2),'go')
        plt.title('final')
        plt.hold('off')
        plt.draw()
        plt.cla()
        time.sleep(2)
        v = np.max(abs(o_a - o_b),0)
        vv = np.median(v)
        sim = np.dot(np.dot((mc_res.size ** 2), math.sqrt(np.max(o_res))), np.mean(n_weight)) / np.maximum((np.dot(ng_samp1, ng_samp2)), 1)
        if edge_core == 1:
            sim = np.dot(sim, 0.5)
    else:
        sim = 0
    print 'sc_cost = ' + str(sc_cost) + '\n'
    print 'sim = ' + str(sim) + '\n'
    return sim, angle, sc_cost

#########TESTING############
##thread = threading.Thread()
##thread.run = do_match
##
##manager = plt.get_current_fig_manager()
##manager.window.after(100, thread.start)
##plt.figure(1)
##plt.show()

