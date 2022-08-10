# -*- coding: utf-8 -*-
'''
Created on Mon Jul 25 14:51:19 2022

@author: Amin
'''

from networkx.algorithms import bipartite
from networkx import convert_matrix
import networkx as nx

from scipy.spatial.distance import cdist
from scipy.linalg import block_diag

import numpy as np

# %%
def randJ_EI_FC(N,J_mean=np.array([[1,2],[1,1.8]])
                ,J_std=np.ones((2,2)),EI_frac=0):
    '''Create random excitatory inhibitory connectivity matrix from input statistics
        
    Args:
        N (integer): Total number of nodes in the network
        J_mean (numpy.ndarray): 2x2 array of the mean for excitatory and inhibitory population connectivity
        J_std (numpy.ndarray): 2x2 array of the standard deviation for excitatory and inhibitory population connectivity
        EI_frac (float): Fraction of excitatory to inhibitory neurons (between 0,1)
    
    Returns:
        array: Randomly generated matrix
    '''
    
    E = round(N*EI_frac)
    I = round(N*(1-EI_frac))
    
    if E > 0:
        J_mean[:,0] = J_mean[:,0]/np.sqrt(E)
        J_std[:,0] = J_std[:,0]/np.sqrt(E)
    if I > 0:
        J_mean[:,1] = -J_mean[:,1]/np.sqrt(I)
        J_std[:,1] = J_std[:,1]/np.sqrt(I)
    
    J = np.zeros((N,N))
    
    J[:E,:E] = np.random.randn(E,E)*J_std[0,0]+J_mean[0,0]
    J[:E,E:] = np.random.randn(E,I)*J_std[0,1]+J_mean[0,1]
    J[E:,:E] = np.random.randn(I,E)*J_std[1,0]+J_mean[1,0]
    J[E:,E:] = np.random.randn(I,I)*J_std[1,1]+J_mean[1,1]
    
    return J
    
# %%
def bipartite_connectivity(M,N,p):
    '''Create random bipartite connectivity matrix
        ref: https://en.wikipedia.org/wiki/Bipartite_graph
        
    Args:
        M (integer): Number of nodes in the first partite
        N (integer): Number of nodes in the second partite
        p (float): Connection probability (between 0,1)
    
    Returns:
        array: Randomly generated matrix
    
    '''
    G = bipartite.random_graph(M, N, p)
    return convert_matrix.to_numpy_array(G)

# %%
def erdos_renyi_connectivity(N,p):
    '''Create random Erdos Renyi connectivity matrix
        ref: https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
        
    Args:
        N (integer): Number of nodes in the network
        p (float): Connection probability (between 0,1)
    
    Returns:
        numpy.ndarray: Randomly generated matrix
    '''
    
    G = nx.erdos_renyi_graph(N,p)
    return convert_matrix.to_numpy_array(G)
# %%
def downstream_uniform_connectivity(M,N,g):
    '''Downstream normal connectivity matrix
        
    Args:
        N (integer): Number of nodes in the network
        g (float): Connection strength
    
    Returns:
        numpy.ndarray: Randomly generated matrix
    '''
    J = g*(.5+.5*np.random.rand(M+N,M+N))
    J[:M,M:]=0
    
    return J


# %%
def normal_connectivity(N,g):
    '''Normal random connectivity matrix
        
    Args:
        N (integer): Number of nodes in the network
        g (float): Connection strength
    
    Returns:
        numpy.ndarray: Randomly generated matrix
    '''
    
    return g*np.random.normal(loc=0.0, scale=1/N, size=(N,N))
    
        
# %%
def dag_connectivity(N,p=.5,g=1.5):
    '''Directed acyclic graph random connectivity matrix
        ref: https://en.wikipedia.org/wiki/Directed_acyclic_graph
        
    Args:
        N (integer): Number of nodes in the network
        p (float): Connection probability (between 0,1) look at the documentation of gnp_random_graph

    Returns:
        numpy.ndarray: Randomly generated matrix
    '''
    
    J = g*np.ones((N,N))*(np.random.rand()<p)
    J *= np.tri(*J.shape,k=-1)
    
    return J
    
# %%
def geometrical_connectivity(
            N,decay=1,EI_frac=0,
            mean=[[0.1838,-0.2582],[0.0754,-0.4243]],
            prob=[[.2,.5],[.5,.5]]
        ):
    '''Create random  connectivity graph that respects the geometry of the nodes in which nodes that are closer are more likely to be connected
        
    Args:
        N (integer): Number of nodes in the network
        decay (float): Decay of the weight strength as a function of physical distance
        EI_frac (float): Fraction of excitatory to inhibitory nodes
        mean (array): 2x2 array representing the mean of the EE/EI/IE/II population
        prob (array): 2x2 array representing the probability of the EE/EI/IE/II population

    Returns:
        numpy.ndarray: Randomly generated matrix
        array: Location of the nodes in the simulated physical space
    '''
    
    def EI_block_diag(cs,vs):
        return np.hstack((
        np.vstack((block_diag(*[np.ones((cs[0,i],cs[0,i]))*vs[0,0,i] for i in range(len(cs[0]))]),
                   block_diag(*[np.ones((cs[1,i],cs[0,i]))*vs[1,0,i] for i in range(len(cs[0]))]))) ,
        np.vstack((block_diag(*[np.ones((cs[0,i],cs[1,i]))*vs[0,1,i] for i in range(len(cs[0]))]),
                   block_diag(*[np.ones((cs[1,i],cs[1,i]))*vs[1,1,i] for i in range(len(cs[0]))])))
        ))
    
    E = round(N*EI_frac)
    I = N - E
    
    X = np.random.rand(N,2)
    
    J_prob = EI_block_diag(np.array([[E],[I]]),np.array(prob)[:,:,np.newaxis])
    J = np.exp(-cdist(X,X)**2/decay)*np.random.binomial(n=1,p=J_prob)
    
    J[:E,:E] = mean[0][0]*J[:E,:E]/J[:E,:E].mean()
    J[:E,E:] = mean[0][1]*J[:E,E:]/J[:E,:E].mean()
    J[E:,:E] = mean[1][0]*J[E:,:E]/J[:E,:E].mean()
    J[E:,E:] = mean[1][1]*J[E:,E:]/J[:E,:E].mean()
    
    return J,X


# %%
def clustered_connectivity(
            N,EI_frac=0,C=10,C_std=[.2,0],
            clusters_mean=[[0.1838,-0.2582],[0.0754,-0.4243]],clusters_stds=[[.0,.0],[.0,.0]],clusters_prob=[[.2,.5],[.5,.5]],
            external_mean=[[.0036,-.0258],[.0094,-.0638]],external_stds=[[.0,.0],[.0,.0]],external_prob=[[.2,.5],[.5,.5]],
            external=None,cluster_size=None
        ):
    '''Create random clustered inhibitory excitatory connectivity graph 
        
    Args:
        N (integer): Number of nodes in the network
        EI_frac (float): Fraction of excitatory to inhibitory nodes
        clusters_mean (array): 2x2 array representing the connection mean for in cluster connections (EE/EI/IE/EE)
        clusters_stds (array): 2x2 array representing the connection standard deviation for in cluster connections
        clusters_prob (array): 2x2 array representing the connection probability for in cluster connections
        external_mean (array): 2x2 array representing the connection mean for out of cluster connections
        external_stds (array): 2x2 array representing the connection standard deviation for out of cluster connections
        external_prob (array): 2x2 array representing the connection probability for out of cluster connections
        external (string): Out of cluster connectivity pattern, choose from ('cluster-block','cluster-column','random')
        cluster_size (array): The number of nodes in each cluster (pre-given)

    Returns:
        numpy.ndarray: Randomly generated matrix
        array: Array of number of nodes in each cluster, first row corresponds to excitatory and second row corresponds to inhibitory
    '''
    
    def EI_block_diag(cs,vs):
        return np.hstack((
                np.vstack((block_diag(*[np.ones((cs[0,i],cs[0,i]))*vs[0,0,i] for i in range(len(cs[0]))]),
                           block_diag(*[np.ones((cs[1,i],cs[0,i]))*vs[1,0,i] for i in range(len(cs[0]))]))) ,
                np.vstack((block_diag(*[np.ones((cs[0,i],cs[1,i]))*vs[0,1,i] for i in range(len(cs[0]))]),
                           block_diag(*[np.ones((cs[1,i],cs[1,i]))*vs[1,1,i] for i in range(len(cs[0]))])))
            ))
    
    
    if cluster_size is None:
        E = round(N*EI_frac)
        I = N - E
        cluster_size = np.round((np.array([[E,I]]).T/C)*np.array(C_std)[:,np.newaxis]*np.random.randn(2,C) + (np.array([[E,I]]).T/C)).astype(int)
        cluster_size[:,-1] = np.array([E,I]).T-cluster_size[:,:-1].sum(1)
    else:
        E,I = cluster_size.sum(1)
    
    c_mean = np.zeros((2,2,C))
    c_prob = np.zeros((2,2,C))
    c_stds = np.zeros((2,2,C))
    
    c_mean[0,:,:] = np.vstack((clusters_mean[0][0]*cluster_size[0,:].mean()/cluster_size[0,:],clusters_mean[0][1]*cluster_size[0,:].mean()/cluster_size[0,:]))
    c_mean[1,:,:] = np.vstack((clusters_mean[1][0]*cluster_size[1,:].mean()/cluster_size[1,:],clusters_mean[1][1]*cluster_size[1,:].mean()/cluster_size[1,:]))
    
    c_prob[0,:,:] = np.vstack((clusters_prob[0][0]+np.zeros((C)),clusters_prob[0][1]+np.zeros((C))))
    c_prob[1,:,:] = np.vstack((clusters_prob[1][0]+np.zeros((C)),clusters_prob[1][1]+np.zeros((C))))
    
    c_stds[0,:,:] = np.vstack((clusters_stds[0][0]+np.zeros((C)),clusters_stds[0][1]+np.zeros((C))))
    c_stds[1,:,:] = np.vstack((clusters_stds[1][0]+np.zeros((C)),clusters_stds[1][1]+np.zeros((C))))
    
    e_size = cluster_size.sum(1)[:,np.newaxis]
    e_prob = np.array(external_prob)[:,:,np.newaxis]
    e_mean = np.array(external_mean)[:,:,np.newaxis]
    e_stds = np.array(external_stds)[:,:,np.newaxis]
    
    JC_prob = EI_block_diag(cluster_size,c_prob)
    JC_mean = EI_block_diag(cluster_size,c_mean)
    JC_stds = EI_block_diag(cluster_size,c_stds)
    
    JC_mask = EI_block_diag(cluster_size,np.ones((2,2,C)))
    
    if external=='cluster-block':
        jc_mask = EI_block_diag(np.ones((2,C),dtype=int),np.ones((2,2,C)))
        je_mean = EI_block_diag(np.array([[C],[C]],dtype=int),e_mean)*(1-jc_mask)
        je_stds = EI_block_diag(np.array([[C],[C]],dtype=int),e_stds)*(1-jc_mask)
        je = (np.random.randn(2*C,2*C)*je_stds+je_mean)*(1-jc_mask)
        JE_mean = je.repeat(cluster_size.flatten(),axis=0).repeat(cluster_size.flatten(),axis=1)
    elif external == 'cluster-column':
        jc_mask = EI_block_diag(np.ones((2,C),dtype=int),np.ones((2,2,C)))
        
        je = np.hstack((
                 np.vstack(((np.random.randn(1,C)*e_stds[0,0]+e_mean[0,0]).repeat(C,axis=0),
                           (np.random.randn(1,C)*e_stds[1,0]+e_mean[1,0]).repeat(C,axis=0))),
                 np.vstack(((np.random.randn(1,C)*e_stds[0,1]+e_mean[0,1]).repeat(C,axis=0),
                           (np.random.randn(1,C)*e_stds[1,1]+e_mean[1,1]).repeat(C,axis=0)))
             ))
        
        JE_mean = je.repeat(cluster_size.flatten(),axis=0).repeat(cluster_size.flatten(),axis=1)
    else:
        JE_mean = EI_block_diag(e_size,e_mean)*(1-JC_mask)

    
    JE_prob = EI_block_diag(e_size,e_prob)*(1-JC_mask)
    JE_stds = EI_block_diag(e_size,e_stds)*(1-JC_mask)
    
    
    J_prob = JC_prob + JE_prob
    J_mean = JC_mean + JE_mean
    J_stds = JC_stds + JE_stds
    
    J = np.random.binomial(n=1,p=J_prob)*(np.random.randn(N,N)*J_stds+J_mean)
    
    J[:E,:E] = np.maximum(0,J[:E,:E])
    J[:E,E:] = np.minimum(0,J[:E,E:])
    J[E:,:E] = np.maximum(0,J[E:,:E])
    J[E:,E:] = np.minimum(0,J[E:,E:])
    
    return J, cluster_size


# %%
def coarse_grain_matrix(J,cluster_size):
    '''Coarse graining a matrix by averaging nodes in the blocks
        
    Args:
        J (numpy.ndarray): Matrix to be coarse grained
        cluster_size (array): Array of sizes to determine the blocks for coarse graining

    Returns:
        numpy.ndarray: Coarse grained matrix
    '''
    
    c_ind = np.hstack((0,np.cumsum(cluster_size)))
    C_J = np.zeros((len(cluster_size),len(cluster_size)))*np.nan
    for i in range(len(cluster_size)):
        for j in range(len(cluster_size)):
            C_J[i,j] = np.nanmean(J[c_ind[i]:c_ind[i+1],:][:,c_ind[j]:c_ind[j+1]])
            
    return C_J