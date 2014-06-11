import scipy.io as sio
import numpy as np
import copy
import matplotlib.pyplot as plt
import pylab as py
import pdb                              #for debugging only
import time
import drawnow

from calc_orient import calc_orient
from hist_cost_2 import hist_cost_2
from dist2 import dist2
from sc_compute import sc_compute


##def call():
##
##        #pdb.set_trace()
##
##        x = sio.loadmat('f1.mat')
##        f1 = {'X':x['f1'][0,0]['X'],'M':x['f1'][0,0]['M'],'O':x['f1'][0,0]['O'],'R':x['f1'][0,0]['R'],'N':x['f1'][0,0]['N'],'RO':x['f1'][0,0]['RO']}
##
##        x = sio.loadmat('f2.mat')
##        f2 = {'X':x['f2'][0,0]['X'],'M':x['f2'][0,0]['M'],'O':x['f2'][0,0]['O'],'R':x['f2'][0,0]['R'],'N':x['f2'][0,0]['N'],'RO':x['f2'][0,0]['RO']}
##
##        [fr,similarity] = register(f1,f2)
##
##        return [fr,similarity]
##
##        

        




def register(f1,f2):

        """Input to this function are two dictionaries f1 and f2. They contain several numpy arrays, possibly having different sizes. Numpy arrays
        in f1 and f2 have keys as: 'X', 'M', 'O', 'R','G','N','RO','OIMG','OREL'. 'OIMG' and 'OREL' are empty, of no use. Output of this code is a list containing 2 dictionaries, fr
        and similarity.""" 
	

	#cla		#to be coded later
        

	m1 = np.mod(f1['M'], np.pi)
	m2 = np.mod(f2['M'], np.pi)		#m1 and m2 are 2-D numpy arrays

	m1s = np.where(m1[:,2] > 3)[0]
	m2s = np.where(m2[:,2] > 3)[0]          #m1s and m2s are 1-D arrays

	singular1 = f1['X'][m1s,:]
	singular2 = f2['X'][m2s,:]

	m_map1 = []
	m_map2 = []

	m1 = f1['M']
	m2 = f2['M']
	X = f1['X']
 	Y = f2['X']
 	oX = f1['O']
 	oY = f2['O']
 	rX = f1['R']
 	rY = f2['R']
 	nX = f1['N']
	nY = f2['N']
	roX = f1['RO']
 	roY = f2['RO']

	ic1  = np.where(f1['M'][:,2] == 5)[0]
	ic2  = np.where(f2['M'][:,2] == 5)[0]              

	low_core = 0
 	if ic1.size > 0 and np.all(f1['M'][ic1,1] > 363):
    		low_core = 1
    		
 	if ic2.size > 0 and np.all(f2['M'][ic2,1] > 363):
    		low_core = 1

    	edge_core = 0

        if ic1.size > 0 and ic2.size > 0:
                GC = 1
        else:
                if singular1.size == 0 or singular2.size == 0:
                        GC = 0
                #core should be below image since we most likely have a huge region without any singularities (cores)i.e Max distance from core to
                #singularity is limited. this idea assumes no heavy noise for core.
                else:
                        if ic1.size > 0 and (np.where(f2['M'][m2s,1] < 300)[0]).size == 0:
                                GC = 3
                        elif ic2.size > 0 and (np.where(f1['M'][m1s,1] < 300)[0]).size == 0:
                                GC = 4
                        else:
                                GC = 2

        

        orients = 100*np.ones([X.shape[0], Y.shape[0]])
        
        start_time = time.time()
                
        for i in range(X.shape[0]):
                for j in range(Y.shape[0]):

                        if m1[i,2] < 5 and m2[j, 2] < 5:
                                orients[i,j] = calc_orient(np.tile(oX[i,:],[1,1]), np.tile(rX[i,:],[1,1]), np.tile(oY[j,:],[1,1]), np.tile(rY[j,:],[1,1]))[0]
                                        #1st call to calc_orient()
        print time.time() - start_time, "seconds"

        #no problem till here

        #pdb.set_trace()


        ndum1 = 0
        ndum2 = 0
        eps_dum = 1
        nsamp1 = X.shape[0]
        nsamp2 = Y.shape[0]
        mean_dist_global = []
        display_flag = 1

        if nsamp2 > nsamp1:
                #as is the case in outlier test
                ndum1 = ndum1 + (nsamp2 - nsamp1)

        #JI: store minimum error in iterations to avoid large jumps away from global minima to local minima

        min_error = -1
        jjj = -1
        region_a = 1
        m1a = []
        m2a = []

        #theta offset after warping from last iteration and the total theta offset from the start (in radians)

        theta_offset_total = 0

        #initialize counter

        k = 1
        s = 1

        #out_vec_{1,2} are indicator vectors for keeping track of estimated outliers on each iteration

        angle = -1
        
        Xo = []
        
        map_new = []       #Since map() is a function name in Python, we use var name map_new instead of map as done in MATLAB code
        o_res = []

        out_vec_1 = [0]*(nsamp1 + 1)
        out_vec_2 = [0]*(nsamp2 + 1)


        sc = -1
        area = -1
        

        i1 = []
        i2 = []

        #print 'nsamp1 = ' + str(nsamp1)
        #print ' nsamp2 = ' + str(nsamp2)

        #print 'Going inside nested for loops: '


        for i in range(nsamp1):
                #pdb.set_trace()
                start_time = time.time()
                print i
                for j in range(nsamp2):
                        
                
                        #print 'i = ' + str(i) + ' j = ' + str(j) + '\n'

                                                
                        t_angle = m1[i,3] - m2[j,3]

                        if abs(t_angle) > np.pi/2.75 or m1[i,2] >= 5 or m2[j,2] >= 5 or orients[i,j] < 0.25:
                                continue

                        sct = 0
                        region_a = 0

                        r_t = m2[j,3] - m1[i,3]
                        m1t = copy.deepcopy(m1[:,3])
                        v = copy.deepcopy(X[:,1])
                        w = copy.deepcopy(X[:,0])
                        Xt = copy.deepcopy(X)
                        Xta = copy.deepcopy(Xt)
                        Xt[:,1] = np.cos(r_t)*v - np.sin(r_t)*w
                        Xt[:,0] = np.sin(r_t)*v + np.cos(r_t)*w

                        diffx = Xt[i,0] - Y[j,0]
                        diffy = Xt[i,1] - Y[j,1]
                        Xt[:,0] = Xt[:,0] - diffx
                        Xt[:,1] = Xt[:,1] - diffy
                        Xtt = 1000*np.ones([nsamp2, 2])
                        Xtt[0:nsamp1, 0:2] = copy.deepcopy(Xt[:,0:2])

                        Xt = copy.deepcopy(Xtt)

                        d1 = np.sqrt(dist2(Xt, Y))              #call to dist2() function

                        

                        if GC > 0:
                                angle_diff = t_angle
                                sing1 = copy.deepcopy(singular1)
                                sing1[:,1] = np.cos(r_t)*singular1[:,1] - np.sin(r_t)*singular1[:,0]
                                sing1[:,0] = np.sin(r_t)*singular1[:,1] + np.cos(r_t)*singular1[:,0]

                                cdist = d1[ic1,ic2]

                                if GC == 1:
                                        if np.all(cdist > 0.18):
                                                continue
                                else:
                                        c_ratio = 0



                        xt = min(Xt[0:nsamp1,1])
                        xb = max(Xt[0:nsamp1,1])
                        xl = min(Xt[0:nsamp1,0])
                        xr = max(Xt[0:nsamp1,0])
                        yt = min(Y[0:nsamp2,1])
                        yb = max(Y[0:nsamp2,1])
                        yl = min(Y[0:nsamp2,0])
                        yr = max(Y[0:nsamp2,0])

                        region_t = max(xt,yt)
                        region_b = min(xb,yb)
                        region_r = min(xr,yr)
                        region_l = max(xl,yl)

                        temp1 = np.intersect1d(np.where(Xt[0:nsamp1,0] > region_l)[0], np.where(Xt[0:nsamp1,0] < region_r)[0])
                        temp2 = np.intersect1d(np.where(Xt[0:nsamp1,1] > region_t)[0], np.where(Xt[0:nsamp1,1] < region_b)[0])
                        ind1 = np.intersect1d(temp1,temp2)

                        temp1 = np.intersect1d(np.where(Y[:,0] > region_l)[0], np.where(Y[:,0] < region_r)[0])
                        temp2 = np.intersect1d(np.where(Y[:,1] > region_t)[0], np.where(Y[:,1] < region_b)[0])
                        ind2 = np.intersect1d(temp1,temp2)


                        if ind1.size == 0 or ind2.size == 0:
                                continue

                        nbins_theta = 10
                        nbins_r = 5
                        r_inner = 0.01
                        r_outer = min(region_b - region_t, region_r - region_l)

                        if xl < -0.2 or xr > 1.25 or (GC == 3 and np.all(yt > f1['X'][ic1,1])) or (GC == 4 and np.all(xt < f2['X'][ic2,1])):
                                continue
                        
                        #Original code starts an if 1 block here which is of no use. We don't do that here.

                        if r_outer >= 1/8:
                                r_outer = 1
                                XX = copy.deepcopy(Xt[ind1,:])
                                XX = np.vstack([XX,np.array([(xl + xr)/2, (xb + xt)/2])])
                                YY = copy.deepcopy(Y[ind2,:])
                                YY = np.vstack([YY,np.array([(xl + xr)/2, (xb + xt)/2])])
                                out_vec_1 = np.zeros([1,ind1.size + 1])
                                out_vec_2 = np.zeros([1, ind2.size + 1])
                                
                                [BH1,mean_dist_1]=sc_compute(XX.transpose(), np.zeros([1,ind1.size+1]),mean_dist_global,nbins_theta,nbins_r,r_inner,r_outer,out_vec_1)
                                                        #1st call to sc_compute()
                                [BH2,mean_dist_2]=sc_compute(YY.transpose(),np.zeros([1,ind2.size+1]),mean_dist_1,nbins_theta,nbins_r,r_inner,r_outer,out_vec_2)
                                                        #2nd call to sc_compute()

                                costmat = hist_cost_2(BH1,BH2,r_inner, r_outer, nbins_theta, nbins_r)
                                                        #1st call to hist_cost_2()

                                sct = costmat[ind1.size, ind2.size]

                                a1 = np.amin(costmat,0)
                                
                                a2 = np.amin(costmat,1)
                                

                                sct_p = np.mean(a1) + np.mean(a2)

                                region_a = (region_b - region_t)*(region_r-region_l)

                                if GC != 1 and sct_p > 0.9 or (sct/region_a > 4.5 and region_a > 0.11 or sct/region_a > 10.9) and GC == 1:
                                        continue

                        for r in range(nsamp1):
                               m1t[r] = abs(np.fmod(m1t[r] - t_angle + np.pi, np.pi))

                        #pdb.set_trace()

                        o_diff = []
                        index = 0
                        t_o=[]
                        t_map=[] 
                        map1 = []
                        map2 = []

                        dists = []

                        for ii in range(Xt.shape[0]):
                                v = np.amin(d1[ii,:])
                                py = np.where(d1[ii,:] == v)[0][0]

                                for jj in range(Y.shape[0]):
                                        v = np.amin(d1[:,jj])
                                        px = np.where(d1[:,jj] == v)[0][0]
                                        
                                        if d1[ii,jj] < 0.05 and m1[ii,2] < 5 and m2[jj, 2] < 5:
                                                try:
                                                        temp = map1.index(ii)           #We do this because map1 is a list, hence np.where() cannot be used
                                                except ValueError:
                                                        map1.append(ii)
                                                try:
                                                        temp = map2.index(jj)           #same for map2
                                                except ValueError:
                                                        map2.append(jj)

                                        if px == ii and py == jj and Xtt[ii,0] < 1000 and d1[ii,jj] < 0.045 and m1[ii,2] < 5 and \
                                           m2[jj,2] < 5 and abs(m1t[ii] - m2[jj,3]) < 0.5:
                                                index = index + 1
                                                o_diff.append(abs(m1t[ii] - m2[jj,3]))
                                                t_o.append(orients[ii, jj]*max(m1[ii,5], m2[jj,5]))
                                                t_map.append([ii,jj])
                                                dists.append(d1[ii,jj])
                                                

                        #pdb.set_trace() #check the values of o_diff, t_o, t_map, dists here with MATLAB code      


                        o = np.square(sum(t_o))/index
                        c = sum(t_o) - sum(o_res)
                        

                        if c > 0 and index > 1:

                                m_map1 = copy.deepcopy(map1)
                                m_map2 = copy.deepcopy(map2)
                                sc = sct
                                area = region_a
                                o_res = copy.deepcopy(t_o)
                                map_new = copy.deepcopy(t_map)
                                #cla not coded, Line 337 of MATLAB code
                                test_n = index
                                angle = t_angle
                                Xtt[nsamp1:nsamp2,0] = 0
                                Xtt[nsamp1:nsamp2,1] = 0
                                Xo = copy.deepcopy(Xtt)
                                i1 = copy.deepcopy(ind1)
                                i2 = copy.deepcopy(ind2)
                                
##                                plt.hold(True)
##                                plt.subplot(2,2,1)
##                                plt.plot(Xtt[:,0],Xtt[:,1],'b+',Y[:,0],Y[:,1],'ro')
##                                if GC == 1:
##                                        plt.plot(Xtt[ic1,0],Xtt[ic1,1],'g+',Y[ic2,1],Y[ic2,1],'go')
##                                plt.title(['i = ' + str(i) + ' j = ' + str(j) + ' n1 = ' + str(nsamp1) + ' n2 = ' + str(nsamp2)])
##                                plt.hold(False)
##                                plt.show()
##                                plt.close('all')
                                #pdb.set_trace()
                
                print time.time() - start_time, "seconds"

        # two nested for loops end here

        #print 'for loops end here'

        s1 = np.where(f1['M'][:,2] < 5)[0].size
        s2 = np.where(f2['M'][:,2] < 5)[0].size

        #print 's1 = '
        #print s1
        #print 's2 = '
        #print s2

        if Xo.size > 0:
                fr = copy.deepcopy(f1)
                fr['M'][:,3] = abs(np.fmod(fr['M'][:,3] - angle + np.pi, np.pi))
                fr['X'] = Xo[0:nsamp1,:]
                similarity = {'map':(np.array(map_new) + 1),'o_res':o_res, 'sc':sc, 'area':area, 'angle':angle,'map1':(np.array(m_map1) + 1), 'map2':(np.array(m_map2) + 1)} 
        else:
                #911 on Line 368 has no meaning !!
                fr = copy.deepcopy(f2)
                similarity = {'map':(np.array(map_new) + 1),'o_res':o_res, 'sc':sc, 'area':area, 'angle':angle,'map1':(np.array(m_map1) + 1), 'map2':(np.array(m_map2) + 1)} 

        return [fr,similarity]

        #Code ends
                                   

        
                                
                                



                                        
                                                           
                                                
                                        
                                        
                                                        
