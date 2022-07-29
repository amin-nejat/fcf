# -*- coding: utf-8 -*-
'''
Created on Mon Jul 25 14:51:59 2022

@author: Amin
'''

import matplotlib.pyplot as plt

from functools import reduce

from scipy.signal import convolve
from scipy.io import savemat


import numpy as np
import scipy as sp

# %%
def continuous_to_spktimes(x,times,threshold):
    '''Conversion of continous signals to spike times for testing the ISI delay embedding framework
        
    Args:
        x (numpy.ndarray): Continous signal (1xT)
        times (numpy.ndarray): Sampling times from the signal
        threshold (float): Threshold used for generating spikes
        
    Returns:
        array: Spike times
    
    '''
    integral = 0
    spktimes = []
    for t in range(len(x)):
        integral = np.nansum([integral,x[t]])
        if integral > threshold:
            integral = 0
            spktimes.append(times[t])
    return np.array(spktimes)

# %%
def spktimes_to_rates(spk,n_bins=100,rng=(-1,1),sigma=.1,method='gaussian',save_data=False,file=None):
    '''Conversion of spike times to rates
        
    Args:
        spk (array): Spike times to be converted to rates
        n_bins (numpy.ndarray): Sampling times from the signal
        rng (float): Threshold used for generating spikes
        sigma (float): std of Gaussian used for smoothing rates
        method (string): Smoothing method, choose from ('gaussian','counts')
        save_data (bool): If True the generated rates will be saved in a mat file
        file (string): File address used for saving the mat file
    Returns:
        numpy.ndarray: Rates that are converted from input spikes
        numpy.ndarray: Bins used for windowing the spikes
    '''
    
    bins = np.linspace(rng[0],rng[1],n_bins)
    rate = np.zeros((n_bins,len(spk)))
    bin_edges = np.linspace(rng[0],rng[1],n_bins+1)
    bin_len = (rng[1]-rng[0])/n_bins
    filt = np.exp(-(np.arange(-3*sigma/bin_len,3*sigma/bin_len,(rng[1]-rng[0])/n_bins)/sigma)**2)
    
    for s in range(len(spk)):
        rate[:,s] = np.histogram(spk[s],bin_edges)[0]
        if method == 'gaussian':rate[:,s] = convolve(rate[:,s],filt,mode='same')
            
    if save_data:
        savemat(file+'.mat',{
                'spk':spk,
                'n_bins':n_bins,
                'rng':rng,
                'sigma':sigma,
                'method':method,
                'bins':bins,
                'rate':rate,
                'bin_edges':bin_edges
            })
            
    return rate,bins

# %%
def divide_clusters(c_range,C=10,C_std=.1):
    '''Divide clusters into smaller groups
        
    Args:
        c_range (array): Array of (start,end) indices of the clusters
        C (integer): Number of smaller groups that we want the clusters to be divided to
        C_std (float): Standard deviation of the resulting subclusters

    Returns:
        array: Sub-clusters
    '''
    c_range_ = []
    for c in c_range:
        c_ = np.round(((c[1]-c[0])/C)*C_std*np.random.randn(C)) + ((c[1]-c[0])/C).astype(int)
        c_[-1] = (c[1]-c[0]) - c_[:-1].sum()
        C_range = np.cumsum(np.hstack((c[0],c_))).astype(int)
        C_range = [(C_range[ci],C_range[ci+1]) for ci in range(len(C_range)-1)]
        c_range_ += list(C_range)
    
    return np.array(c_range_)

# %%
def aggregate_spikes(spk,ind):
    '''Aggregate spikes from multiple channels
        
    Args:
        spk (array): Array of (channel,spike_time)
        ind (array): Indices of the nodes that we want to aggregte their spikes

    Returns:
        array: Aggregated spikes
    '''
    
    if isinstance(spk[0], np.ndarray):
        return [reduce(np.union1d, tuple([spk[i].tolist() for i in ind_])) for ind_ in ind]
    else:
        return [reduce(np.union1d, tuple([spk[i] for i in ind_])) for ind_ in ind]

# %%
def unsort(spk,ind=None,sample_n=3,ens_n=10,ens_ind=None,save_data=False,file=None):
    if ens_ind is None:
        ens_ind = [[]]*ens_n
        
    ens_spk = [[]]*ens_n
    for ens in range(ens_n):
        if len(ens_ind[ens])==0:
            ens_ind[ens] = [np.random.choice(ind_,size=sample_n,replace=False).astype(int) for ind_ in ind]
        ens_spk[ens] = aggregate_spikes(spk,ens_ind[ens])
        
    if save_data: savemat(file+'.mat',{'ens_ind':ens_ind})
        
    return ens_ind,ens_spk

# %%
def sequential_recording(X,rates,t,fov_sz,visualize=True,save=False,file=None):
    '''Mask data according to a sequential recording experiment where the recording FOV moves sequentially to cover the space
        
    Args:
        X (numpy.ndarray): Matrix to be coarse grained
        C_size (array): Array of sizes to determine the blocks for coarse graining
        X: Locations of the nodes in the network
        rates: Activities of the nodes in the network
        t: Time for the sampled rates
        fov_sz: Size of the field of view (FOV) for sequential recording
        visualize (bool): If true the graph will be visualized
        save (bool): If True the plot will be saved
        file (string): File address for saving the plot

    Returns:
        numpy.ndarray: Masked rates according to the simulated sequential recording experiment
        array: Array of indices of the nodes in each ensemble (nodes in the same FOV)
        array: Array of timing of the nodes in the same ensemble
    '''
    
    min_sz = X.min(0).copy()
    max_sz = X.max(0).copy()
    cur_sz = X.min(0).copy()
    
    ens = []
    
    rates_masked = rates.copy()*np.nan
    
    dir_ = 1
    
    if visualize:
        plt.subplots(figsize=((max_sz[1]-min_sz[1])/5,(max_sz[0]-min_sz[0])/5))
        plt.scatter(X[:,1],X[:,0])
    
        IND = [str(i) for i in range(X.shape[0])]
        for i, txt in enumerate(IND):
            plt.annotate(txt, (X[i,1], X[i,0]))

    plt.grid('on')
    
    while True:
        ind = np.logical_and.reduce([ (X[:,0]>=cur_sz[0]),
                        (X[:,1]>=cur_sz[1]),
                        (X[:,0]< cur_sz[0]+fov_sz[0]),
                        (X[:,1]< cur_sz[1]+fov_sz[1])])
        
        if len(ind) > 0:
            ens.append(np.where(ind)[0])
            
        
        if visualize:
            rectangle = plt.Rectangle((cur_sz[1],cur_sz[0]),fov_sz[1],fov_sz[0],fc=[0,0,0,0],ec='red')
            plt.gca().add_patch(rectangle)
        
        if cur_sz[1] >= max_sz[1]:
            cur_sz[1] -= fov_sz[1]
            cur_sz[0] = X[ens[-1],0].max()-2
            dir_ = -1
        elif cur_sz[1] < min_sz[1]:
            cur_sz[1] += fov_sz[1]
            cur_sz[0] = X[ens[-1],0].max()-2
            dir_ = 1
        else:
            if dir_ == 1:
                cur_sz[1] = X[ens[-1],1].max()-2
            else:
                cur_sz[1] = X[ens[-1],1].min()+2
                
        if len(ens) > 1 and len(reduce(np.union1d,ens)) == X.shape[0]:
            break
    
    bins = np.linspace(min(t)-.1,max(t)+.1,len(ens))
    bin_edges = np.linspace(min(t)-.1,max(t)+.1,len(ens)+1)
    ens_t = [(bin_edges[i],bin_edges[i+1]) for i in range(len(bins))]
    for i in range(len(ens)):
        a = rates_masked[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:]
        a[:,ens[i]] = rates[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:][:,ens[i]].copy()
        rates_masked[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:] = a
    
    if visualize:
        if save:
            plt.savefig(file+'.eps',format='eps')
            plt.savefig(file+'.png',format='png')
            plt.savefig(file+'.pdf',format='pdf')
            plt.close('all')
        else:
            plt.show()
    
    return rates_masked,ens,ens_t
