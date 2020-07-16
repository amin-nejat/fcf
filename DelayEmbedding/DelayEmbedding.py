# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin

"""

import numpy as np
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors

def create_delay_vector_spikes(spktimes,dim):
    return np.array([np.append(spktimes[i-dim:i]-spktimes[i-dim-1:i-1],spktimes[i-dim]) for i in range(dim+1,len(spktimes))])

def create_delay_vector(sequence,delay,dim):
     # sequence is a time series corresponding to a single node but can be multidimensional 
     # e.g. 100000x2 (each column of sequence is a time series corresponding to one of the node's coordinates)
     # delay is the delay time
     # dim is the embedding dimension

     duration=len(sequence)
     shift= delay*(dim-1) #omit the +1 because the index numeration starts from 0
     ntrails = duration-shift

     sequence=np.squeeze(sequence)
     if len(np.shape(sequence))==1:
         sequence=np.reshape(sequence, (duration, 1))

     sequenceDim=np.shape(sequence)[1]
     trail = np.zeros((ntrails,sequenceDim*dim))
     vec = lambda x: np.ndarray.flatten(x)
     #vec flattens a matrix by concatenating all its rows into a single row
     for idx in range(ntrails):
         trail[idx,:] = \
         vec(sequence[idx:shift+idx+1:delay,:])
         #note: shift+idx+1 has the final +1 because the last index of the slice gets excluded otherwise
     return trail #in the output trail, as in the input sequence, the row index is time . 

#    def random_projection(x,dim):
#    P =  np.random.rand(np.shape(x)[1],dim)
#    projected = arrayfun(@(i) x[:,:,i]*P, 1:size(x,3), 'UniformOutput',false)
#    projected = cat(3,projected{:})
#     return projected

def cov2corr(cov):
    diag = np.sqrt(np.diag(cov))[:,np.newaxis]
    corr = np.divide(cov,diag@diag.T)
    return corr

        
def reconstruct(cues,lib_cues,lib_targets,n_neighbors=3,n_tests="all"):

    # lib_cues has dimensions L x d1 (where L is a large integer)
    # lib_targets has dimensions L x d2
    # cue has dimension N x d1 (where N could be one or larger)
    
    
    nCues,dimCues=np.shape(cues)
    dimTargets=np.shape(lib_targets)[1]


    if n_tests == None:
        n_tests = nCues
        
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(lib_cues)    
    distances, indices = nbrs.kneighbors(cues)
    # distances is a matrix of dimensions N x k , where k = nNeighbors
    # indices is also a matrix of dimensions N x k , where k = nNeighbors

    weights = np.exp(-distances)
    # the matrix weight has dimensions N x k 
    # we want to divide each row of the matrix by the sum of that row.
    weights = weights/weights.sum(axis=1)[:,None] #using broadcasting 
    # still, weight has dimensions N x k 

    reconstruction=np.zeros((nCues,dimTargets))
    # We want to comput each row of reconstruction by through a weighted sum of vectors from lib_targets
    for idx in range(nCues):
        reconstruction[idx,:] = weights[idx,:]@lib_targets[indices[idx,:],:]
    # The product of a Nxk matrix with a kxd2 matrix

    return reconstruction
     

def interpolate_delay_vectros(delay_vectors,times,kind='nearest'):
    interpolated = np.zeros((len(times), delay_vectors.shape[1]))
    interpolated[:,-1] = times
    
    
    interp = interpolate.interp1d(delay_vectors[:,-1],delay_vectors[:,:-1].T,kind=kind,bounds_error=False)
    interpolated[:,:-1] = interp(times).T
    
    return interpolated
    
def mean_covariance(trials):
    _, nTrails, trailDim = np.shape(trials)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.cov(trials[:,:,idx]))

    corrcoef = np.nanmean(np.array(corrcoefs),0)
    
    return corrcoef

def mean_correlations(trials):
    _, nTrails, trailDim = np.shape(trials)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trials[:,:,idx]))

    corrcoef = np.nanmean(np.array(corrcoefs),0)
    
    return corrcoef
    
def sequential_correlation(trails1,trails2):
    # both trails have size T x d 
    
    # centering = lambda trails: trails--diag(mean(trails,2))*ones(size(trails)); 
    # goodIndices=find(isnan(sum(trails1,1)+sum(trails2,1))==0);

    nTrails, trailDim=np.shape(trails1)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trails1[:,idx], trails2[:,idx])[0,1])

    corrcoef=np.nanmean(np.array(corrcoefs))
    
    return corrcoef

def sequential_mse(trails1,trails2):
    # both trails have size T x d 
    
    # centering = lambda trails: trails--diag(mean(trails,2))*ones(size(trails)); 
    # goodIndices=find(isnan(sum(trails1,1)+sum(trails2,1))==0);

    nTrails, trailDim=np.shape(trails1)

    mses = []
    for idx in range(trailDim):
        mses.append(np.nanmean((trails1[:,idx]-trails2[:,idx])**2))

    mses=np.nanmean(np.array(mses))
    
    return mses

def connectivity(X,test_ratio=.02,delay=10,dim=3,n_neighbors=3,method='corr'):
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,dim,delay)[:,:,np.newaxis], X.T)),2)
    
    n_trails = delay_vectors.shape[0]
    shift = delay*(dim-1)
    
    test_size = np.max([1.0,np.min([np.floor(test_ratio*n_trails),n_trails-shift-1.0])]).astype(int)

    start = n_trails-test_size
    end = n_trails
    test_indices = np.arange(start,end)
    margin = delay*(dim-1)
    train_indices = np.arange(0,start-margin)
    
    nns = build_nn(delay_vectors,train_indices,test_indices,test_ratio=test_ratio,n_neighbors=n_neighbors)
    
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    reconstruction_error = np.zeros((X.shape[1],X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(X.shape[1]):
            reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
            
            if method == 'corr':
                reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
            elif method == 'mse':
                reconstruction_error[i,j] = sequential_mse(reconstruction, targets[:,:,j])
    return reconstruction_error.T-reconstruction_error
    
def build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=3):
    nns = []    
    for i in range(X.shape[2]):
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X[train_indices,:,i])
        distances, indices = nbrs.kneighbors(X[test_indices,:,i])
        weights = np.exp(-distances/distances.sum())
        weights = weights/weights.sum(axis=1)[:,None]
        nns.append((weights,indices))
    return nns

def reconstruction_accuracy(x,y,test_ratio=.02,delay=1,dims=np.array([3]),n_neighbors=3,method='corr'):
    
    delay_x = create_delay_vector(x,delay,dims.max()+1)
    delay_y = create_delay_vector(x,delay,dims.max()+1)
    
    n_trails = delay_x.shape[0]
    shift = delay*(dims.max()-1)
    
    test_size = np.max([1.0,np.min([np.floor(test_ratio*n_trails),n_trails-shift-1.0])]).astype(int)
    
    correlations = np.zeros(dims.shape)
    
    for idx,dim in enumerate(dims):
        start = n_trails-test_size
        end = n_trails
        test_indices = np.arange(start,end)
        margin = delay*(dim-1)
        train_indices = np.arange(0,start-margin)
        
        
        recon = reconstruct(delay_x[test_indices,:dim], \
                            delay_x[train_indices,:dim], \
                            delay_y[train_indices,:dim], \
                            n_neighbors=n_neighbors, n_tests=test_size)
        
        if method == 'corr':
            correlations[idx] = sequential_correlation(recon, delay_y[test_indices,:dim])
        elif method == 'mse':
            correlations[idx] = sequential_mse(recon, delay_y[test_indices,:dim])
        
        
    return correlations.max()