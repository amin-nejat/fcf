# -*- coding: utf-8 -*-
'''
Created on Mon Jul 25 14:53:18 2022

@author: Amin
'''

from scipy.io import savemat
from scipy import interpolate

import numpy as np

# %%
def stimulation_protocol(
        c_range,time_st,time_en,N,n_record,stim_d,rest_d,feasible,amplitude,repetition=1,fraction_stim=.8,save=False,file=None
    ):
    '''Create random stimulation protocol for nodes of a network given some input statistics
        
    Args:
        c_range (array): Array of (start,end) indices of the groups of nodes from which we want to select a subset to stimulate simultaneously
        time_st (float): Start time
        time_en (float): End time
        N (integer): Total number of nodes in the network
        n_record (integer): Number of nodes that will be randomly selected to record from during the stimultion experiment
        stim_d (float): Duration of the stimulation
        rest_d (float): Duration of resting after each stimulation (choose relative to stim_d)
        feasible (array): Boolean array determining which neurons are feasible to record from
        amplitude (float): Strength of the stimulation
        repetition (integer): Number of the repetition of stimulation per node
        fraction_stim (float): Fraction of nodes to be stimulated in each node group
        fontsize (integer): Font size for plotting purposes
        visualize (bool): If True the resulting matrix will be plotted
        save (bool): If True the plot will be saved
        file (string): File address for saving the plot and data
        save_data (bool): If True the generated stimulation protocol information will be saved in a mat file

    Returns: I,t_stim,recorded,stimulated
        I (numpy.ndarray): Stimulation pattern represented as a matrix (NxT) where N is the number of nodes and T is the number of time points, the elements of the matrix correspond to stimulation strength
        t_stim (numpy.ndarray): Timing in which the stimulation is sampled
        recorded (array): Feasible node indices in the network that are selected to record from (based on the input 'feasible' criterion)
        stimulated (array): Stimulated node indices
    '''
    
    t_stim = np.linspace(time_st,time_en,repetition*int((stim_d+rest_d)/stim_d)*(len(c_range)))
    I = np.zeros((len(t_stim),N))
    
    stimulated = []
    for c in range(len(c_range)):
        sz = int(((c_range[c][1]-c_range[c][0])*fraction_stim).round())
        rand_sample = np.random.choice(np.arange(c_range[c][0],c_range[c][1]),
               size=sz,replace=False).astype(int)
        stimulated.append(rand_sample)
        
    recorded = []
    for c in range(len(c_range)):
        rand_sample = stimulated[c]
        rand_sample = rand_sample[np.where(feasible[rand_sample])[0]]
        recorded += rand_sample[:n_record].tolist()
    
    for r in range(repetition):
        clusters = np.arange(len(c_range))
        np.random.shuffle(clusters)
        
        for c_idx,c in enumerate(clusters):
            time_idx = r*len(clusters)+c_idx
            rand_sample = stimulated[c]
            d1 = 1
            d2 = int((stim_d+rest_d)/stim_d)
            I[d2*time_idx+d2//2:d2*time_idx+d2//2+d1,rand_sample] = amplitude[rand_sample]
    
    if save:
        savemat(file+'.mat',{
                'c_range':c_range,
                'time_st':time_st,
                'time_en':time_en,
                'N':N,
                'n_record':n_record,
                'stim_d':stim_d,
                'rest_d':rest_d,
                'feasible':feasible,
                'amplitude':amplitude,
                'repetition':repetition,
                'fraction_stim':fraction_stim,
                'I':I,
                't_stim':t_stim,
                'recorded':recorded,
                'stimulated':stimulated
            })
        
    inp = interpolate.interp1d(t_stim,I.T,kind='nearest',bounds_error=False)

    return I,t_stim,recorded,stimulated,inp
