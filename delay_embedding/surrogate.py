# -*- coding: utf-8 -*-
'''
Created on Fri Jul 22 17:16:13 2022

@author: Amin
'''

from sklearn.neighbors import NearestNeighbors
import numpy as np

import ray

# %%
@ray.remote
def twin_surrogates(X,N):
    '''Create twin surrogates for significance evaluation and hypothesis testing
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate the embedding dimension
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    '''
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
    d, indices = nbrs.kneighbors(X)
    threshold = np.percentile(d[:,1],10)
    print('Twin Surrogate Threshold: ' + str(threshold))
    
    nbrs = NearestNeighbors(radius=threshold, algorithm='ball_tree').fit(X)
    d, indices = nbrs.radius_neighbors(X)
    indices = [list(i) for i in indices]
    u,a = np.unique(indices,return_inverse=True)
    ind = [u[a[i]] for i in range(len(a))]
    eln = [len(i) for i in ind]
    surr = np.zeros((N,X.shape[0]))
    L = X.shape[0]
    for sn in range(N):
        kn=np.random.randint(0,L,1)[0]-1
        for j in range(L):
            kn += 1
            surr[sn,j] = X[kn,0]
            kn = ind[kn][np.random.randint(0,eln[kn],1)[0]]
            if kn==L-1:
                kn=L//2
    
    return surr

