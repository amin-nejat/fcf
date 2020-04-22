# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin

"""

import numpy as np
from sklearn.neighbors import NearestNeighbors

def create_delay_vector(sequence,delay,sz):
     # sequence is a time series corresponding to a single node but can be multidimensional 
     # e.g. 100000x2 (each column of sequence is a time series corresponding to one of the node's coordinates)
     # delay is the delay time
     # sz is the embedding dimension

     duration=len(sequence)
     shift= delay*(sz-1) #omit the +1 because the index numeration starts from 0
     ntrails = duration-shift

     sequence=np.squeeze(sequence)
     if len(np.shape(sequence))==1:
         sequence=np.reshape(sequence, (duration, 1))

     sequenceDim=np.shape(sequence)[1]
     trail = np.zeros((ntrails,sequenceDim*sz))
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

def reconstruct(cues,lib_cues,lib_targets,nNeighbors=3,nTests="all"):

    # lib_cues has dimensions L x d1 (where L is a large integer)
    # lib_targets has dimensions L x d2
    # cue has dimension N x d1 (where N could be one or larger)
    
    
    nCues,dimCues=np.shape(cues)
    dimTargets=np.shape(lib_targets)[1]


    if nTests==None:
        nTests=nCues
        
    nbrs = NearestNeighbors(nNeighbors, algorithm='ball_tree').fit(lib_cues)    
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
     
       
def sequentialCorr(trails1,trails2):
    # both trails have size T x d 
    
    # centering = lambda trails: trails--diag(mean(trails,2))*ones(size(trails)); 
    # goodIndices=find(isnan(sum(trails1,1)+sum(trails2,1))==0);

    nTrails, trailDim=np.shape(trails1)

    corrcoefs=np.zeros(trailDim)
    for idx in range(trailDim):
        corrcoefs[idx]=np.corrcoef(trails1[:,idx], trails2[:,idx])[0,1]

    corrcoef=np.nanmean(corrcoefs)
    return corrcoef
