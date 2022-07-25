# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:13:42 2022

@author: Amin
"""

import numpy as np

# %%
def cov2corr(cov):
    '''Transform covariance matrix to correlation matrix
    
    Args:
        cov (numpy.ndarray): Covariance matrix (NxN)
    Returns:
        numpy.ndarray: Correlation matrix (NxN)
    '''
    
    diag = np.sqrt(np.diag(cov))[:,np.newaxis]
    corr = np.divide(cov,diag@diag.T)
    return corr

# %%    
def mean_covariance(trials):
    '''Compute mean covariance matrix from delay vectors
    
    Args:
        trials (np.ndarray): delay vectors
        
    Returns:
        np.ndarray: Mean covariance computed from different dimensions
            in the delay coordinates
    '''
    
    _, nTrails, trailDim = np.shape(trials)

    covariances = []
    for idx in range(trailDim):
        covariances.append(np.cov(trials[:,:,idx]))

    covariances = np.nanmean(np.array(covariances),0)
    
    return covariances

# %%
def mean_correlations(trials):
    '''Compute mean correlation matrix from delay vectors
    
    Args:
        trials (numpy.ndarray): delay vectors
        
    Returns:
        numpy.ndarray: Mean correlation computed from different dimensions
            in the delay coordinates
    '''
    
    _, nTrails, trailDim = np.shape(trials)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trials[:,:,idx]))

    corrcoef = np.nanmean(np.array(corrcoefs),0)
    
    return corrcoef
    
# %%
def sequential_correlation(trails1,trails2):
    '''Compute the correlation between two signals from their delay representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean correlation between the two delay representations
    
    '''
    
    nTrails, trailDim = np.shape(trails1)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trails1[:,idx], trails2[:,idx])[0,1])

    corrcoef=np.nanmean(np.array(corrcoefs))
    
    return corrcoef

# %%
def sequential_mse(trails1,trails2):
    '''Compute the mean squared error between two signals from their delay 
        representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean squared error between the two delay representations
    
    '''
    
    nTrails, trailDim=np.shape(trails1)

    mses = []
    for idx in range(trailDim):
        mses.append(np.nanmean((trails1[:,idx]-trails2[:,idx])**2))

    mses=np.nanmean(np.array(mses))
    
    return mses

# %%
def correlation_FC(X,transform='fisher'):
    T, N = X.shape
            
    # Loop over each pair and calculate the correlation between signals i and j
    correlation_mat  = np.zeros((N,N))*np.nan
    for i in range(N):
        for j in range(i,N):
            cc = np.corrcoef(X[:,i],X[:,j])[0,1]
            correlation_mat[i,j] = cc
            correlation_mat[j,i] = cc
            
    # Apply transformation   
    if transform == 'fisher':
        correlation_mat = np.arctanh(correlation_mat)
        
    return correlation_mat
