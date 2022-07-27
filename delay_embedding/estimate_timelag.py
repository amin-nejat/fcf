# -*- coding: utf-8 -*-
'''
Created on Fri Jul 22 17:15:48 2022

@author: Amin
'''
import ray
import numpy as np
from sklearn.metrics import mutual_info_score

# %%
@ray.remote
def _timelag_autocorr(X, time_delay):
    return np.corrcoef(X[:-time_delay].flatten(),X[time_delay:].flatten())[0,1]

def timelag_autocorr(X, max_time_delay=20):
    refs = [_timelag_autocorr.remote(X, time_delay) for time_delay in np.arange(1,max_time_delay)]
    return ray.get(refs)

# %%
@ray.remote
def _timelag_mutinfo(X, time_delay, n_bins):
    '''Calculate the mutual information given the time delay.'''
    contingency = np.histogram2d(X[:-time_delay].flatten(), X[time_delay:].flatten(),bins=n_bins)[0]
    mutual_information = mutual_info_score(None, None, contingency=contingency)
    return mutual_information

def timelag_mutinfo(X,max_time_delay,n_bins=100):
    refs = [_timelag_mutinfo.remote(X, time_delay, n_bins) for time_delay in np.arange(1,max_time_delay)]
    return ray.get(refs)

# %%
def estimate_timelag(X,max_time_delay,method='autocorr'):
    '''Estimate the embedding time lag from the data
    
    Args:
        X (numpy.ndarray): (TxN) multivariate signal for which we want to estimate the embedding time lag
        method (string): Method for estimating the embedding time lag tau, choose from ('autocorr', 'mutinf')
    Returns:
        integer: Estimated embedding time lag
    '''
    if method == 'autocorr':
        acorr = np.array(timelag_autocorr(X,max_time_delay))
        sign_change = np.where(np.logical_and(acorr[:-1]>=0, acorr[1:]<0))
        return sign_change[0][0]-1 if len(sign_change) > 0 else max_time_delay-1
    if method == 'mutinf':
        mutinfo = timelag_mutinfo(X,max_time_delay)
        return np.argmin(mutinfo)+1