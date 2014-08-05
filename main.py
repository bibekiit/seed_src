from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os, glob, pdb, threading
import matplotlib.pyplot as plt
from calc_EER import calc_EER
from do_match import do_match
#
#    Author: Bibek Behera
#    Email: bibek.iitkgp@gmail.com
#    Description: Main script to perform FVC2002 matching experiment
#

def main():
    BASE_IMG = ''
    os.chdir('C:\\Users\\mooc-20\\Downloads\\seed_src - Copy\\MATLAB\\Db1_B')

    file_names = np.array(glob.glob('*.tif'))

    o_start = 0

    if len(BASE_IMG) != 0:
        for i in range(1,file_names.size+1):
            if len(file_names[i-1]) != len(BASE_IMG):
                continue
            if (file_names[i-1] == BASE_IMG):
                o_start = i
                break

        i_start = o_start + 1

        for i in range(o_start+1,file_names.size+1):
            if len(file_names[i-1]) != len(COMP_IMG):
                continue
            if (file_names[i-1] == COMP_IMG):
                i_start = i
                break
    else:
        o_start = 1
        i_start = 2

    if i_start == o_start + 1 and o_start == 1:
        RES_G = np.array([])
        RES_B = np.array([])
        VG = np.array([])
        VB = np.array([])
        SC_G = np.array([])
        SC_B = np.array([])
        same_flag = 0
        fnmc = 0
        fmc = 0
        tmc = 0
        tnmc = 0
        print 'Starting from beginning'
        avg_fnm_err = 0
        avg_tnm_err = 0
        avg_tm_err = 0
        avg_fm_err = 0

    error_boundary = 0.5

    for index1 in range(o_start,(file_names.size)+1):
        file_a = file_names[index1-1]
        count_G = 0
        count_B = 0
        BASE_IMG = file_a
        for index2 in range(i_start,file_names.size+1):
            COMP_IMG = file_names[index2-1]
            if len(COMP_IMG) == len(BASE_IMG) and (COMP_IMG[ 0: len(COMP_IMG) - 6] == BASE_IMG[ 0: len(BASE_IMG) - 6]):
                same_flag = 1
            else:
                same_flag = 0
                if np.mod(index1, 8) != 1 or np.mod(index2, 8) != 1:
                    same_flag = - 1
            if same_flag > - 1:
                print 'Comparing ' + file_names[index1-1] + ' with ' + file_names[index2-1] + ' same_flag = ' + str(same_flag)
                res, vv, sc = do_match(file_names[index1-1], file_names[index2-1]) # nargout=3

            if res < error_boundary and same_flag == 1:
                print 'Bad Genuine'
                print ' ' + file_names[index1-1] + ' with ' + file_names[index2-1] + ' same_flag = ' + str(same_flag)
                # pause

            if res > error_boundary and same_flag == 0:
                print 'Bad Impostor'
                print ' ' + file_names[index1-1] + ' with ' + file_names[index2-1] + ' same_flag = ' + str(same_flag)
                # pause

            if same_flag == 1:
                RES_G = np.insert(RES_G, count_G, res)
                VG = np.insert(VG, count_G, vv)
                SC_G = np.insert(SC_G, count_G, sc)
                count_G = count_G + 1
                if res > error_boundary:
                    tmc = tmc + 1
                    avg_tm_err = (np.dot(avg_tm_err, (tmc - 1)) + res) / tmc
                else:
                    fnmc = fnmc + 1
                    avg_fnm_err = (np.dot(avg_fnm_err, (fnmc - 1)) + res) / fnmc
            else:
                if same_flag == 0:
                    RES_B = np.insert(RES_B, count_B, res)
                    SC_B = np.insert(SC_B, count_B, sc)
                    VB = np.insert(VB, count_B, vv)
                    count_B = count_B + 1
                    if res <= error_boundary:
                        tnmc = tnmc + 1
                        avg_tnm_err = (np.dot(avg_tnm_err, (tnmc - 1)) + res) / tnmc
                    else:
                        fmc = fmc + 1
                        avg_fm_err = (np.dot(avg_fm_err, (fmc - 1)) + res) / fmc
            COMP_IMG = file_names[index2-1]

        plt.clf
        i_start = index1 + 2
        ig, ib = calc_EER(1.0 / (RES_G - np.dot(0.3, SC_G) + 0.5), 1.0 / (RES_B - np.dot(0.3, SC_B) + 0.5)) # nargout=2
        plt.hold('on')
        plt.plot(ig, 'b')
        plt.plot(ib, 'r')
        plt.hold('off')
        plt.axis([1040, 1060, 0.4, 0.8])
        plt.xlabel('threshold (t)')
        plt.ylabel('EER (\n %)')
        plt.draw()

#########TESTING############
thread = threading.Thread()
thread.run = main

manager = plt.get_current_fig_manager()
manager.window.after(100, thread.start)
plt.figure(1)
plt.show()

test
