from __future__ import division
import numpy as np
from scipy.io import loadmat,savemat
import os

def calc_EER(genuine_dists, imposter_dists):
    count = 0
    genuine_scores = 1 / (genuine_dists)
    imposter_scores = 1 / (imposter_dists)
    gen_rate = np.array([])
    imp_rate = np.array([])
    #genuine_scores =  genuine_dists;
    for threshold in np.linspace(-10,0.01, 11001):
        tempvec = np.flatnonzero((genuine_scores < threshold))
        Pgen = tempvec.size / genuine_scores.size
        count = count + 1
        gen_rate = np.insert(gen_rate, (count -1), np.dot(Pgen, 100))
        tempvec = np.flatnonzero((imposter_scores >= threshold))
        if imposter_scores.size > 0:
            Pimp = tempvec.size / imposter_scores.size
            imp_rate = np.insert(imp_rate, (count -1), np.dot(Pimp, 100))
        else:
            Pimp = 0
            imp_rate = 0
    return gen_rate, imp_rate
