# -*- coding: utf-8 -*-
'''
Created on Wed Jan 15 10:39:40 2020
 
@author: ff215, Amin

'''
from sklearn.neighbors import NearestNeighbors

from scipy.io import savemat
from scipy import stats
import numpy as np


from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S

import ray
# %%
@ray.remote
def remote_connectivity(X,**args):
    return connectivity(X,**args)[0]

# %%
def connectivity(X,test_ratio=.02,delay=10,dim=3,n_neighbors=4,mask=None,transform='fisher',return_pval=False,n_surrogates=20,save=False,file=None):
    '''
    
    Args:
        X (numpy.ndarray): Multivariate signal to compute functional connectivity from (TxN), columns are the time series for different chanenls/neurons/pixels
        test_ratio (float): Fraction of the test/train split (between 0,1)
        delay (integer): Delay embedding time delay 
        dim (integer): Delay embedding dimensionality
        mask (numpy.ndarray): 2D boolean array represeting which elements of the functional connectivity matrix we want to compute
        transform (string): Transofrmation applied to the inferred functional connectivity, choose from ('fisher','identity')
        return_pval (bool): If true the pvales will be computed based on twin surrogates method
        n_surrogates (integer): Number of twin surrogates datasets created for computing the pvalues
        save (bool): If True the results of the computations will be saved in a mat file
        file (string): File address in which the results mat file will be saved
        
    Returns:
        numpy.ndarray: the output is a matrix whose i-j entry (i.e. reconstruction_error[i,j]) is the error level observed when reconstructing channel i from channel, which used as the surrogate for the functional connectivity
        numpy.ndarray: If return_pval is True this function also returns the matrix of pvalues
    '''
    T, N = X.shape
    tShift = delay*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
    delay_vectors = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    # How much data are we going to try and reconstruct?
    tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

    # Get indices for training and test datasets
    iStart = tDelay - tTest; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    
    # Calculate reconstruction error only for elements we want to compute
    if mask is None: mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    # Build nearest neighbour data structures for multiple delay vectors
    # nns is a list of tuples: the first being the weights used for the forecasting technique, 
    # and the second element the elements corresponding to the training delay vector
    nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio,n_neighbors)
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]
    
    # Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    # This will be quantified based on the correlation coefficient between the true and forecast signals
    reconstruction_error = np.zeros((N,N))*np.nan
    for i, j in zip(*mask_idx):
        # Use k-nearest neigbors forecasting technique to estimate the forecast signal
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
        reconstruction_error[i,j] = E.sequential_correlation(reconstruction, targets[:,:,j])
    
    # Apply transformation   
    if transform == 'fisher': fcf = np.arctanh(reconstruction_error)
    if transform == 'identity': fcf = reconstruction_error
    

    if return_pval:
        refs = [S.twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
        surrogates = np.array(ray.get(refs))
        refs = [remote_connectivity.remote(
                surrogates[:,i].T,
                test_ratio=test_ratio,
                delay=delay,
                dim=dim,
                n_neighbors=n_neighbors,
                mask=mask,
                transform=transform
            ) for i in range(surrogates.shape[1])]
        fcf_surrogates = np.stack(ray.get(refs))
        print(fcf_surrogates.shape)
        
        # Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval = 1-2*np.abs(np.array([[stats.percentileofscore(fcf_surrogates[:,i,j],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

    else:
        pval,surrogates = None,None
        
        
    if save:
        savemat(file+'.mat',{
            'fcf':fcf,
            'pval':pval,
            'surrogates':surrogates,
            'n_surrogates':n_surrogates,
            'test_ratio':test_ratio,
            'delay':delay,
            'dim':dim,
            'n_neighbors':n_neighbors,
            'mask':mask,
            'transform':transform
        })

    return fcf, pval, surrogates

# %%
def build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=4):
    '''Build nearest neighbour data structures for multiple delay vectors
    
    Args:
        X (numpy.ndarray): 3D (TxDxN) numpy array of delay vectors for multiple signals
        train_indices (array): indices used for the inference of the CCM mapping
        test_indices (array): indices used for applying the inferred CCM mapping and further reconstruction
    
    Returns:
        array: Nearest neighbor data structures (see the documentation of NearestNeighbors.kneighbors)
    '''
    
    nns = []    
    for i in range(X.shape[2]):
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X[train_indices,:,i])
        distances, indices = nbrs.kneighbors(X[test_indices,:,i])
        weights = np.exp(-distances)
        weights = weights/(weights.sum(axis=1)[:,np.newaxis])
        nns.append((weights,indices))
    return nns

