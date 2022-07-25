# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:15:27 2022

@author: Amin
"""
from sklearn.neighbors import NearestNeighbors

import numpy as np
import re

# %%
def dim_hilbert(X, tau):
    L = len(X)
    asig=np.fft.fft(X)
    asig[np.ceil(0.5+L/2):]=0
    asig=2*np.fft.ifft(asig)
    esig=[asig.real(), asig.imag()]
    return esig

# %%
def dim_fnn(X, tau, method, RT=20, mm=0.01):
    L = len(X)
    spos = [m.start() for m in re.finditer('-',method)]
    
    if len(spos)==1:
        RT=float(method[spos:])
        
    if len(spos)==2:
        RT=float(method[spos[0]+1:spos[1]])
        mm=float(method[spos[1]+1:])
    
    pfnn = [1]
    d = 1
    esig = X.copy()
    while pfnn[-1] > mm:
        nbrs = NearestNeighbors(2, algorithm='ball_tree').fit(esig[:,:-tau].T)
        NNdist, NNid = nbrs.kneighbors(esig[:,:-tau].T)
        
        NNdist = NNdist[:,1:]
        NNid = NNid[:,1:]
        
        d=d+1
        EL=L-(d-1)*tau
        esig=np.zeros((d*X.shape[0],EL))
        for dn in range(d):
            esig[dn*X.shape[0]:(dn+1)*X.shape[0],:]=X[:,(dn)*tau:L-(d-dn-1)*tau].copy()
        
        # Checking false nearest neighbors
        FNdist = np.zeros((EL,1))
        for tn in range(esig.shape[1]):
            FNdist[tn]=np.sqrt(((esig[:,tn]-esig[:,NNid[tn,0]])**2).sum())
        
        pfnn.append(len(np.where((FNdist**2-NNdist**2)>((RT**2)*(NNdist**2)))[0])/EL)
        
    D = d-1 
    esig=np.zeros((D*X.shape[0],L-(D-1)*tau))
    
    for dn in range(D):
        esig[dn*X.shape[0]:(dn+1)*X.shape[0],:]=X[:,dn*tau:L-(D-dn-1)*tau].copy()
    
    return D,esig,pfnn

# %%
def estimate_dimension(X, tau, method='fnn'):
    '''Estimate the embedding dimension from the data
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate
            the embedding dimension
        tau (integer): Taken time delay 
        method (string): Method for estimating the embedding dimension, choose
            from ('fnn', hilbert)
    Returns:
        integer: Estimated embedding dimension
    '''
    # TODO: Implement correlation dimension and box counting dimension methods
    
    if method == 'hilbert':
        return dim_hilbert(X, tau)

    if 'fnn' in method:
        return dim_fnn(X, tau)


