# -*- coding: utf-8 -*-
'''
Created on Fri Jul 22 17:15:27 2022

@author: Amin
'''
from sklearn.neighbors import NearestNeighbors

import numpy as np
import ray

from delay_embedding import helpers

# %%
@ray.remote
def _dim_fnn(X, time_delay, dimension):
    '''Calculate the number of false nearest neighbors in a certain embedding dimension, based on heuristics.
    '''
    X_embedded = helpers.create_delay_vector(X,delay=time_delay,dim=dimension)

    neighbor = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X_embedded)
    distances, indices = neighbor.kneighbors(X_embedded)
    distance = distances[:,1]
    X_first_nbhrs = X[indices[:,1]]

    epsilon = 2. * np.std(X)
    tolerance = 10

    neg_dim_delay = - dimension * time_delay
    distance_slice = distance[:neg_dim_delay]
    X_rolled = np.roll(X, neg_dim_delay)
    X_rolled_slice = slice(len(X) - len(X_embedded), neg_dim_delay)
    X_first_nbhrs_rolled = np.roll(X_first_nbhrs, neg_dim_delay)

    neighbor_abs_diff = np.abs(
        X_rolled[X_rolled_slice] - X_first_nbhrs_rolled[:neg_dim_delay]
    )
    
    false_neighbor_ratio = neighbor_abs_diff/distance_slice[:,None]
    
    false_neighbor_criteria = false_neighbor_ratio > tolerance
    limited_dataset_criteria = distance_slice < epsilon
    n_false_neighbors = np.sum(false_neighbor_criteria * limited_dataset_criteria[:,None])
    
    return n_false_neighbors

def dim_fnn(X, time_delay, max_dimension=20):
    refs = [_dim_fnn.remote(X, time_delay, dimension) for dimension in np.arange(2,max_dimension)]
    return ray.get(refs)

# %%
def estimate_dimension(X, tau, method='fnn'):
    '''Estimate the embedding dimension from the data
    
    Args:
        X (np.array): (T,N) multivariate signal for which we want to estimate the embedding dimension
        tau (integer): Taken time delay 
        method (string): Method for estimating the embedding dimension, choose from ('fnn', 'hilbert')
    
    Returns:
        integer: Estimated embedding dimension
    '''
    # TODO: Implement correlation dimension and box counting dimension methods
    
    if method == 'hilbert':
        raise NotImplementedError()

    if 'fnn' in method:
        return np.argmin(dim_fnn(X, tau))+2


