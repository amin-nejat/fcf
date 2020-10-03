# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin

"""
import re
import numpy as np
from scipy.io import savemat
from scipy import interpolate
from sklearn.neighbors import NearestNeighbors
#from functools import reduce
from scipy import stats
from scipy import sparse

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


def connectivity(X,test_ratio=.02,delay=10,dim=3,n_neighbors=3,method='corr',mask=None, transform='fisher', return_pval=False, n_surrogates=20, save_data=False, file=None):

    """
    the input X is a matrix whose columns are the time series for different chanenls. 
    the output is a matrix whose i-j entry (i.e. reconstruction_error[i,j]) is the error level 
    observed when reconstructing channel i from channel j.
     
    """
    
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    n_trails = delay_vectors.shape[0]
    shift = delay*(dim-1)
    
    test_size = np.max([1.0,np.min([np.floor(test_ratio*n_trails),n_trails-shift-1.0])]).astype(int)

    start = n_trails-test_size
    end = n_trails
    test_indices = np.arange(start,end)
    margin = delay*(dim-1)
    train_indices = np.arange(0,start-margin)
    
    if mask is None:
        mask = np.zeros((delay_vectors.shape[2],delay_vectors.shape[2])).astype(bool)
    
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio=test_ratio,n_neighbors=n_neighbors)
    nns = [[]]*delay_vectors.shape[2]
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]
    
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    reconstruction_error = np.zeros((X.shape[1],X.shape[1]))*np.nan
    
    pairs = [(mask_idx[0][i],mask_idx[1][i]) for i in range(len(mask_idx[0]))]
    for pair in pairs:
        i,j = pair
        
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])
        if method == 'corr':
            reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
        elif method == 'mse':
            reconstruction_error[i,j] = sequential_mse(reconstruction, targets[:,:,j])
        
    
    if return_pval:
        surrogates = np.array(list(map(lambda x: twin_surrogates(x,n_surrogates), delay_vectors.transpose([2,0,1]))))
        connectivity_surr = np.zeros((X.shape[1],X.shape[1],n_surrogates))
        for n in range(n_surrogates):
            connectivity_surr[:,:,n] = connectivity(surrogates[:,n,:].T,test_ratio=.1,delay=delay,dim=dim)
        pval = 1-2*np.abs(np.array([[stats.percentileofscore(connectivity_surr[i,j,:],reconstruction_error[i,j],kind='strict') 
                for j in range(X.shape[1])] for i in range(X.shape[1])])/100 - .5)
        
        if transform == 'fisher':
            reconstruction_error = np.arctanh(reconstruction_error)
            
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error,'pval':pval,'surrogates':surrogates,'connectivity_surr':connectivity_surr})
        
        return reconstruction_error, pval
    else:
        if transform == 'fisher':
            reconstruction_error = np.arctanh(reconstruction_error)
        
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error})
            
        return reconstruction_error
    
def build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=3):
    nns = []    
    for i in range(X.shape[2]):
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X[train_indices,:,i])
        distances, indices = nbrs.kneighbors(X[test_indices,:,i])
        weights = np.exp(-distances)#/(np.median(distances)))
        weights = weights/(weights.sum(axis=1)[:,np.newaxis])
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


def estimate_dimension(X, tau, method='fnn'):
    # Estimate the embedding dimension
    L = X.shape[1]
    
    if method == 'hilbert':
        asig=np.fft.fft(X)
        asig[np.ceil(0.5+L/2):]=0
        asig=2*np.fft.ifft(asig)
        esig=[asig.real(), asig.imag()]
    elif 'fnn' in method:
        RT=20
        mm=0.01
        
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
            
            # Check false nearest neighbors
            FNdist = np.zeros((EL,1))
            for tn in range(esig.shape[1]):
                FNdist[tn]=np.sqrt(((esig[:,tn]-esig[:,NNid[tn,0]])**2).sum())
            
            pfnn.append(len(np.where((FNdist**2-NNdist**2)>((RT**2)*(NNdist**2)))[0])/EL)
            
        D = d-1 
        esig=np.zeros((D*X.shape[0],L-(D-1)*tau))
        
        for dn in range(D):
            esig[dn*X.shape[0]:(dn+1)*X.shape[0],:]=X[:,dn*tau:L-(D-dn-1)*tau].copy()
            
        return D,esig
#    elif ~isempty(strfind(method,'corrdim')):
#        cdim=1; d=0;
#        while cdim(end)+1>d:
#            d=d+1
#            EL=L-(d-1)*tau
#            esig=zeros(d,EL)
#            for dn in range(d): 
#                esig[dn,:]=sig[1+(dn-1)*tau:L-(d-dn)*tau]
#                
#            cdim.append(corrdim(esig,[],EstAlg)); # correlation dimension
    
#    elif ~isempty(strfind(method,'boxdim')):
#        bdim=1; 
#        d=0;
#        while bdim(end)+1>d:
#            d=d+1
#            EL=L-(d-1)*tau
#            esig=np.zeros((d,EL)) 
#            for dn in range(d):
#                esig[dn,:]=sig[1+(dn-1)*tau:L-(d-dn)*tau]
#            bdim.append(boxdim(esig,[],q,EstAlg))


def estimate_timelag(X,method='autocorr'):
    # Estimate the embedding time-lag
    L = len(X)
    if method == 'autocorr':
        x=np.arange(len(X)).T
        FM = np.ones((len(x),4))
        for pn in range(1,4):
            CX=x**pn
            FM[:,pn]=(CX-CX.mean())/CX.std()
        
        csig=X-FM@(np.linalg.pinv(FM)@X)
        acorr = np.real(np.fft.ifft(np.abs(np.fft.fft(csig))**2).min(1))
        tau = np.where(np.logical_and(acorr[:-1]>=0, acorr[1:]<0))[0][0]
        
    elif method == 'mutinf':
        NB=np.round(np.exp(0.636)*(L-1)**(2/5)).astype(int)
        ss=(X.max()-X.min())/NB/10 # optimal number of bins and small shift
        bb=np.linspace(X.min()-ss,X.max()+ss,NB+1)
        bc=(bb[:-1]+bb[1:])/2
        bw=np.mean(np.diff(bb)) # bins boundaries, centers and width
        mi=np.zeros((L))*np.nan; # mutual information
        for kn in range(L-1):
            sig1=X[:L-kn]
            sig2=X[kn:L]
            # Calculate probabilities
            prob1=np.zeros((NB,1))
            bid1=np.zeros((L-kn)).astype(int)
            prob2=np.zeros((NB,1))
            bid2=np.zeros((L-kn)).astype(int)
            jprob=np.zeros((NB,NB))
            
            for tn in range(L-kn):
                cid1=np.floor(0.5+(sig1[tn]-bc[0])/bw).astype(int)
                bid1[tn]=cid1
                prob1[cid1]=prob1[cid1]+1
                cid2=np.floor(0.5+(sig2[tn]-bc[0])/bw).astype(int)
                bid2[tn]=cid2
                prob2[cid2]=prob2[cid2]+1
                jprob[cid1,cid2]=jprob[cid1,cid2]+1
                jid=(cid1,cid2)
                
            prob1=prob1/(L-kn)
            prob2=prob2/(L-kn)
            jprob=jprob/(L-kn)
            prob1=prob1[bid1]
            prob2=prob2[bid2]
            jprob=jprob[jid[0],jid[1]]
            
            # Estimate mutual information
            mi[kn]=np.nansum(jprob*np.log2(jprob/(prob1*prob2)))
            # Stop if minimum occured
            if kn>0 and mi[kn]>mi[kn-1]:
                tau=kn
                break
    
    return tau


def twin_surrogates(X,N):
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
    


def nn_surrogates(X,N,n_neighbors=2):
    nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X)
    d, indices = nbrs.kneighbors(X)
    surr = np.zeros((N,X.shape[0]))
    for sn in range(N):
        node = 0
        for j in range(X.shape[0]):
            surr[sn,j] = X[node,0]
            node = indices[node+1,np.random.randint(0,n_neighbors,1)[0]]
            if node == X.shape[0]-1:
                if n_neighbors == 1:
                    node = indices[-1,0]-1
                else:
                    node = indices[-1,np.random.randint(0,n_neighbors-1,1)[0]+1]
            
            
    return surr

