# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:11:39 2022

@author: Amin
"""
from sklearn.neighbors import NearestNeighbors
from scipy import interpolate
import numpy as np

# %% delay embedding helpers
def create_delay_vector_spikes(spktimes,dim):
    '''Create ISI delay vectors from spike times
    
    Args:
        spktimes (numpy.array): Array of spike times for a single channel
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded spike train
    '''
    return np.array([np.append(spktimes[i-dim:i]-spktimes[i-dim-1:i-1],spktimes[i-dim]) for i in range(dim+1,len(spktimes))])

# %%
def create_delay_vector(sequence,delay,dim):
    '''Create delay vectors from rate or sequence data
    
    Args:
        sequence (numpy.ndarray): Time series (TxN) corresponding to a single node 
            but can be multidimensional
        delay (integer): Delay used in the embedding (t-delay,t-2*delay,...)
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded sequence
    '''
    
    T = sequence.shape[0]   #Number of time-points we're working with
    tShift = delay*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Make sure vector is 2D
    sequence = np.squeeze(sequence)
    if len(np.shape(sequence))==1:
        sequence = np.reshape(sequence, (T,1))

    #Number of neurons 
    N = sequence.shape[1]

    #Preallocate delay vectors
    dvs = np.zeros((tDelay,N*dim)) #[length of delay vectors x # of delay vectors]

    #Create fn to flatten matrix if time series is multidimensional
    vec = lambda x: np.ndarray.flatten(x)

    #Loop through delay time points
    for idx in range(tDelay):
        # note: shift+idx+1 has the final +1 because the last index of the slice gets excluded otherwise
        dvs[idx,:] = vec(sequence[idx:tShift+idx+1:delay,:]) 
    
    return dvs 

# %%
def random_projection(x,dim):
    '''Random projection of delay vectors for a more isotropic representation
    
    Args:
        x (numpy.ndarray): Delay coordinates of a sequence (n,time,delay)
        dim (integer): Projection dimensionality
        
    Returns:
        numpy.ndarray: Random projected signals
    '''
    P =  np.random.rand(np.shape(x)[1],dim)
    projected = np.array([x[:,:,i]*P for i in range(x.shape[2])]).transpose(1,2,0)
    return projected

# %%
def reconstruct(cues,lib_cues,lib_targets,n_neighbors=3,n_tests='all'):
    '''Reconstruct the shadow manifold of one time series from another one
        using Convergent Cross Mapping principle and based on k-nearest-neighbours
        method
    
    Args:
        lib_cues (numpy.ndarray): Library of the cue manifold use for 
            reconstruction, the dimensions are L x d1 (where L is a large integer),
            the library is used for the inference of the CCM map
        lib_targets (numpy.ndarray): Library of the target manifold to be used
            for the reconstruction of the missing part, the dimensions are
            L x d2, the library is used for the inference of the CCM map
        cue (numpy.ndarray): The corresponding part in the cue manifold to the
            missing part of the target manifold, cue has dimension N x d1 
            (where N could be one or larger)
        
    Returns:
        numpy.ndarray: Reconstruction of the missing parts of the target manifold
    
    '''
    
    nCues,dimCues=np.shape(cues)
    dimTargets=np.shape(lib_targets)[1]


    if n_tests == None:
        n_tests = nCues
        
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(lib_cues)
    
    # distances is a matrix of dimensions N x k , where k = nNeighbors
    # indices is also a matrix of dimensions N x k , where k = nNeighbors
    distances, indices = nbrs.kneighbors(cues)

    # the matrix weight has dimensions N x k 
    # we want to divide each row of the matrix by the sum of that row.
    weights = np.exp(-distances)
    
    # still, weight has dimensions N x k 
    weights = weights/weights.sum(axis=1)[:,None] #using broadcasting 
    
    # We want to comput each row of reconstruction by through a weighted sum of vectors from lib_targets
    reconstruction=np.zeros((nCues,dimTargets))
    for idx in range(nCues):
        reconstruction[idx,:] = weights[idx,:]@lib_targets[indices[idx,:],:] # The product of a Nxk matrix with a kxd2 matrix

    return reconstruction

# %%
def interpolate_delay_vectors(delay_vectors,times,kind='nearest'):
    '''Interpolte delay vectors used for making the spiking ISI delay 
        coordinates look more continuous
    
    Args:
        delay_vectors (numpy.ndarray): 3D (N,time,delay) numpy array of the 
            delay coordinates
        times (numpy.ndarray): The time points in which delay vectors are sampled
        kind (string): Interpolation type (look at interp1d documentation)
        
    Returns:
        numpy.ndarray: Interpolated delay vectors
    '''
    
    interpolated = np.zeros((len(times), delay_vectors.shape[1]))
    interpolated[:,-1] = times
    
    
    interp = interpolate.interp1d(delay_vectors[:,-1],delay_vectors[:,:-1].T,kind=kind,bounds_error=False)
    interpolated[:,:-1] = interp(times).T
    
    return interpolated