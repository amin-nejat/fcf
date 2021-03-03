# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:29:24 2020

@author: Amin
"""
from operator import itemgetter
from itertools import groupby
from scipy.io import savemat
from copy import deepcopy
from scipy import stats
import numpy as np

# %%

def interventional_connectivity(activity,stim,t=None,bin_size=10,skip_pre=10,skip_pst=4,pval_threshold=1,methods=['mean_isi','aggr_ks','mean_ks','aggr_ks_pval'],save_data=False,file=None):
    """Create point clouds from a video using Matching Pursuit or Local Max algorithms
    
    Args:
        activity (numpy.ndarray): 2D (N,T) numpy array of signals in the 
            perturbed state
        stim (array): Array of tuples of channel, stimulation start, and 
            stimulation end [(chn_1,str_1,end_1),(chn_1,str_,end_1),...]
        t (numpy.ndarray): If the activity is rates instead of spiking the 
            timing of the sampled signals is given in t
        bin_size (float): Window size used for binning the activity and 
            computing pre vs. post stimulation firing distribution
        skip_pre (float): How much time to skip before the stimulation for pre
            distribution
        skip_pst (float): How much time to skip before the stimulation for pre
            distribution
        pval_threshold (float): A float between 0 and 1 determining significance
            threshold for computing interventional connectivity
        methods (array): Which metrics to use for the interventional 
            connectivity; the output of this function is an array with each 
            element corresponding to one metric given in this array            
        save_data (bool): If True the computed values will be saved in a mat file
        file (string): Name of the file used to save the mat file
    
    Returns:
        dict: Dictionary with interventional connectivity matrices evaluated 
            for each given input metric (methods)
    
    """
    stim_ = deepcopy(stim)
    
    for i in range(len(stim)):
        if t is None:
            pst_isi = [np.diff(activity[j][(activity[j] <  stim_[i][2]+skip_pst+bin_size) & (activity[j] >= stim_[i][2]+skip_pst)]) for j in range(len(activity))]
            pre_isi = [np.diff(activity[j][(activity[j] >= stim_[i][1]-skip_pre-bin_size) & (activity[j] <  stim_[i][1]-skip_pre)]) for j in range(len(activity))]
        else:
            pst_isi = [activity[j][(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)] for j in range(len(activity))]
            pre_isi = [activity[j][(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)] for j in range(len(activity))]
        
        stim_[i] += (pre_isi,pst_isi)
    
    stim_g = [(k, [(x3,x4) for _,x1,x2,x3,x4 in g]) for k, g in groupby(sorted(stim_,key=itemgetter(0)), key=itemgetter(0))]
    
    output = {}
    count = {}
    for m in methods:
        output[m] = np.zeros((len(activity), len(activity)))*np.nan
        count[m] = np.zeros((len(activity), len(activity)))*.0
    
    for i in range(len(stim_g)): # stimulation channel
        print('Computing intervention effect for channel ' + str(i))
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = []
            aggr_pst_isi = []
            for j in range(len(stim_g[i][1])): # stimulation event
                if 'mean_ks' in methods:
                    if len(stim_g[i][1][j][0][n]) > 0 and len(stim_g[i][1][j][1][n]) > 0:
                        ks,p = stats.mstats.ks_2samp(stim_g[i][1][j][0][n],stim_g[i][1][j][1][n])
                        if p <= pval_threshold:
                            output['mean_ks'][stim_g[i][0],n] = np.nansum((output['mean_ks'][stim_g[i][0],n],ks))
                            count['mean_ks'][stim_g[i][0],n] += 1
                            
                if 'mean_isi' in methods:
                    df_f = (stim_g[i][1][j][1][n].mean()-stim_g[i][1][j][0][n].mean())
                    output['mean_isi'][stim_g[i][0],n] = np.nansum((output['mean_isi'][stim_g[i][0],n],df_f))
                    count['mean_isi'][stim_g[i][0],n] += 1
                
                aggr_pre_isi.append(stim_g[i][1][j][0][n])
                aggr_pst_isi.append(stim_g[i][1][j][1][n])
            
            if 'aggr_ks' in methods:
                if np.array(aggr_pre_isi).size > 0 and np.array(aggr_pst_isi).size > 0:
                    ks,p = stats.mstats.ks_2samp(np.hstack(aggr_pre_isi),np.hstack(aggr_pst_isi))
                    if p <= pval_threshold:
                        output['aggr_ks'][stim_g[i][0]][n] = ks
                        count['aggr_ks'][stim_g[i][0]][n] = 1
                        if 'aggr_ks_pval' in methods:
                            output['aggr_ks_pval'][stim_g[i][0]][n] = p
                            count['aggr_ks_pval'][stim_g[i][0]][n] = 1
        for m in methods:
            output[m] /= count[m]
    
    if save_data:
        savemat(file+'.mat',{'activity':activity,'stim':stim,'t':t,'bin_size':bin_size,
                             'skip_pre':skip_pre,'skip_pst':skip_pst,'pval_threshold':pval_threshold,
                             'methods':methods,'output':output})
            
    return output

