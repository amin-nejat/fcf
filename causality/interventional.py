# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 13:29:24 2020

@author: Amin
"""
from operator import itemgetter
from itertools import groupby
from copy import deepcopy

from scipy import stats

import numpy as np
import os
# %% helpers
def average_treatment_effect(pre,pst):
    '''mean post stimulation and pre stimulation difference
    '''
    cnn,pvalue = np.nan,np.nan
    for i in range(len(pre)):
        df_f = abs(pst[i].mean()-pre[i].mean())
        cnn = np.nansum((cnn,df_f))
    return cnn/len(pre),pvalue

def aggregated_kolmogorov_smirnov(pre,pst):
    '''ks distribution distance between pre and post stimulation activity
    '''
    if np.array(pre).size > 0 and np.array(pst).size > 0:
        ks,p = stats.mstats.ks_2samp(np.hstack(pre),np.hstack(pst))
        cnn = ks
        pvalue = p
    else:
        cnn,pvalue = np.nan,np.nan
    return cnn,pvalue

def mean_kolmogorov_smirnov(pre,pst):
    '''mean ks distribution distance between pre and post stimulation activity 
    computer on individual instances of stimulation
    '''
    cnn,pvalue = np.nan,np.nan
    for i in range(len(pre)):
        if len(pre[i]) > 0 and len(pst[i]) > 0:
            ks,p = stats.mstats.ks_2samp(pre[i],pst[i])
            cnn = np.nansum((cnn,ks))
            pvalue = np.nansum((pvalue,p))
    return cnn/len(pre),pvalue/len(pre)

# %%
def interventional_connectivity(
        activity,stim,mask=None,t=None,
        bin_size=10,skip_pre=10,skip_pst=4,
        method='aggr_ks',
        save=False,load=False,file=None
    ):
    '''Compute interventional connectivity by measure statistical difference between pre 
    and post stimulation activity
    
    Args:
        activity (np.ndarray): (N,T) np array of signals in the perturbed state
        stim (array): Array of tuples of channel, stimulation start, and end time [(chn_i,str_i,end_i)]
        t (np.ndarray): If the activity is rates instead of spiking the timing of the sampled signals is given in t
        bin_size (float): Time window used for binning the activity and computing pre vs. post stimulation firing distribution
        skip_pre (float): Time to skip before the stimulation for pre distribution
        skip_pst (float): Time to skip before the stimulation for pre distribution
        method (string): Metrics for interventional connectivity: aggr_ks, mean_isi, mean_ks
        save_data (bool): If True the computed values will be saved in a mat file
        file (string): Name of the file used to save the mat file
    
    Returns:
        cnn: Dinterventional connectivity matrix evaluated for each given input metric
        pvalue: corresponding significance
    '''
    
    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']

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
    
    cnn = np.zeros((len(activity), len(activity)))*np.nan
    pvalue = np.zeros((len(activity), len(activity)))*np.nan
    
    for i in range(len(stim_g)): # stimulation channel
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = [stim_g[i][1][j][0][n] for j in range(len(stim_g[i][1]))]
            aggr_pst_isi = [stim_g[i][1][j][1][n] for j in range(len(stim_g[i][1]))]
            
            if method == 'mean_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = mean_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
            if method == 'mean_isi':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = average_treatment_effect(aggr_pre_isi,aggr_pst_isi)
            if method == 'aggr_ks':
                cnn[stim_g[i][0]][n],pvalue[stim_g[i][0]][n] = aggregated_kolmogorov_smirnov(aggr_pre_isi,aggr_pst_isi)
                            
        
    if mask is None: mask = np.zeros((len(activity),len(activity))).astype(bool)
    
    cnn = cnn.T
    pvalue = pvalue.T
    
    cnn[mask] = np.nan
    pvalue[mask] = np.nan
    
    if save: np.save(file,{'cnn':cnn,'pvalue':pvalue})
    
    return cnn,pvalue

