# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020
 
@author: ff215, Amin

"""
from sklearn.neighbors import NearestNeighbors
from multiprocessing import Pool
from functools import partial
from scipy import interpolate
from scipy.io import savemat
from scipy import stats
import numpy as np
import re

def create_delay_vector_spikes(spktimes,dim):
    """Create ISI delay vectors from spike times
    
    Args:
        spktimes (numpy.array): Array of spike times for a single channel
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded spike train
    
    """
    
    return np.array([np.append(spktimes[i-dim:i]-spktimes[i-dim-1:i-1],spktimes[i-dim]) for i in range(dim+1,len(spktimes))])

def create_delay_vector(sequence,delay,dim):
    """Create delay vectors from rate or sequence data
    
    Args:
        sequence (numpy.ndarray): Time series (TxN) corresponding to a single node 
            but can be multidimensional
        delay (integer): Delay used in the embedding (t-delay,t-2*delay,...)
        dim (integer): Embedding dimensionality
        
    Returns:
        numpy.ndarray: Delay coordinates of the embedded sequence
    
    """
    
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

def random_projection(x,dim):
    """Random projection of delay vectors for a more isotropic representation
    
    Args:
        x (numpy.ndarray): Delay coordinates of a sequence (n,time,delay)
        dim (integer): Projection dimensionality
        
    Returns:
        numpy.ndarray: Random projected signals
    
    """
    P =  np.random.rand(np.shape(x)[1],dim)
    projected = np.array([x[:,:,i]*P for i in range(x.shape[2])]).transpose(1,2,0)
    return projected

def cov2corr(cov):
    """Transform covariance matrix to correlation matrix
    
    Args:
        cov (numpy.ndarray): Covariance matrix (NxN)
        
    Returns:
        numpy.ndarray: Correlation matrix (NxN)
    
    """
    
    diag = np.sqrt(np.diag(cov))[:,np.newaxis]
    corr = np.divide(cov,diag@diag.T)
    return corr

        
def reconstruct(cues,lib_cues,lib_targets,n_neighbors=3,n_tests="all"):
    """Reconstruct the shadow manifold of one time series from another one
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
    
    """
    
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
     

def interpolate_delay_vectors(delay_vectors,times,kind='nearest'):
    """Interpolte delay vectors used for making the spiking ISI delay 
        coordinates look more continuous
    
    Args:
        delay_vectors (numpy.ndarray): 3D (N,time,delay) numpy array of the 
            delay coordinates
        times (numpy.ndarray): The time points in which delay vectors are sampled
        kind (string): Interpolation type (look at interp1d documentation)
        
    Returns:
        numpy.ndarray: Interpolated delay vectors
    
    """
    
    interpolated = np.zeros((len(times), delay_vectors.shape[1]))
    interpolated[:,-1] = times
    
    
    interp = interpolate.interp1d(delay_vectors[:,-1],delay_vectors[:,:-1].T,kind=kind,bounds_error=False)
    interpolated[:,:-1] = interp(times).T
    
    return interpolated
    
def mean_covariance(trials):
    """Compute mean covariance matrix from delay vectors
    
    Args:
        trials (numpy.ndarray): delay vectors
        
    Returns:
        numpy.ndarray: Mean covariance computed from different dimensions
            in the delay coordinates
    
    """
    
    _, nTrails, trailDim = np.shape(trials)

    covariances = []
    for idx in range(trailDim):
        covariances.append(np.cov(trials[:,:,idx]))

    covariances = np.nanmean(np.array(covariances),0)
    
    return covariances

def mean_correlations(trials):
    """Compute mean correlation matrix from delay vectors
    
    Args:
        trials (numpy.ndarray): delay vectors
        
    Returns:
        numpy.ndarray: Mean correlation computed from different dimensions
            in the delay coordinates
    
    """
    
    _, nTrails, trailDim = np.shape(trials)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trials[:,:,idx]))

    corrcoef = np.nanmean(np.array(corrcoefs),0)
    
    return corrcoef
    
def sequential_correlation(trails1,trails2):
    """Compute the correlation between two signals from their delay representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean correlation between the two delay representations
    
    """
    
    nTrails, trailDim = np.shape(trails1)

    corrcoefs = []
    for idx in range(trailDim):
        corrcoefs.append(np.corrcoef(trails1[:,idx], trails2[:,idx])[0,1])

    corrcoef=np.nanmean(np.array(corrcoefs))
    
    return corrcoef

def sequential_mse(trails1,trails2):
    """Compute the mean squared error between two signals from their delay 
        representations
    
    Args:
        trails1 (numpy.ndarray): delay vectors of the first signal (shape Txd)
        trails2 (numpy.ndarray): delay vectors of the second signal (shape Txd)
        
    Returns:
        float: Mean squared error between the two delay representations
    
    """
    
    nTrails, trailDim=np.shape(trails1)

    mses = []
    for idx in range(trailDim):
        mses.append(np.nanmean((trails1[:,idx]-trails2[:,idx])**2))

    mses=np.nanmean(np.array(mses))
    
    return mses

def connectivity(X,test_ratio=.02,delay=10,dim=3,n_neighbors=4,method='corr',mask=None,transform='fisher',return_pval=False,n_surrogates=20,save_data=False,file=None,parallel=False,MAX_PROCESSES=96):
    """Create point clouds from a video using Matching Pursuit or Local Max algorithms
    
    Args:
        X (numpy.ndarray): Multivariate signal to compute functional 
            connectivity from (TxN), columns are the time series for 
            different chanenls/neurons/pixels
        test_ratio (float): Fraction of the test/train split (between 0,1)
        delay (integer): Delay embedding time delay 
        dim (integer): Delay embedding dimensionality
        method (string): Method used for computing the reconstructability,,
            choose from ('mse','corr')
        mask (numpy.ndarray): 2D boolean array represeting which elements of the 
            functional connectivity matrix we want to compute
        transform (string): Transofrmation applied to the inferred functional 
            connectivity, choose from ('fisher','identity')
        return_pval (bool): If true the pvales will be computed based on twin
            surrogates method
        n_surrogates (integer): Number of twin surrogates datasets created for 
            computing the pvalues
        save_data (bool): If True the results of the computations will be saved 
            in a mat file
        file (string): File address in which the results mat file will be saved
        parallel (bool): If True the computations are done in parallel
        MAX_PROCESSES (integer): Max number of processes instantiated for parallel 
            processing
        
    Returns:
        numpy.ndarray: the output is a matrix whose i-j entry (i.e. reconstruction_error[i,j]) 
            is the error level observed when reconstructing channel i from channel, 
            which used as the surrogate for the functional connectivity
        numpy.ndarray: If return_pval is True this function also returns the 
            matrix of pvalues
    """
    T, N = X.shape
    tShift = delay*(dim-1)  #Max time shift
    tDelay = T - tShift     #Length of delay vectors

    #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
    delay_vectors = np.concatenate(list(map(lambda x: create_delay_vector(x,delay,dim)[:,:,np.newaxis], X.T)),2)
    
    #How much data are we going to try and reconstruct?
    tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)

    #Get indices for training and test datasets
    iStart = tDelay - tTest; iEnd = tDelay
    test_indices = np.arange(iStart,iEnd)
    train_indices = np.arange(0,iStart-tShift)
    targets = delay_vectors[test_indices,:,:]
    lib_targets = delay_vectors[train_indices,:,:]
    
    #Calculate reconstruction error only for elements we want to compute
    if mask is None:
        mask = np.zeros((N,N)).astype(bool)
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    #Build nearest neighbour data structures for multiple delay vectors
    #nns is a list of tuples: the first being the weights used for the forecasting technique, 
    #and the second element the elements corresponding to the training delay vector
    nns_ = build_nn(delay_vectors[:,:,mask_u_idx],train_indices,test_indices,test_ratio,n_neighbors)
    nns = [[]]*N
    for i,idx in enumerate(mask_u_idx):
        nns[idx] = nns_[i]
    
    #Loop over each pair and calculate the topological embeddedness for signal i by another signal j
    #This will be quantified based on the correlation coefficient (or MSE) between the true and forecast signals
    reconstruction_error = np.zeros((N,N))*np.nan
    for i, j in zip(*mask_idx):

        #Use k-nearest neigbors forecasting technique to estimate the forecast signal
        reconstruction = np.array([nns[i][0][idx,:]@lib_targets[nns[i][1][idx,:],:,j] for idx in range(len(test_indices))])

        if method == 'corr':
            reconstruction_error[i,j] = sequential_correlation(reconstruction, targets[:,:,j])
        elif method == 'mse':
            reconstruction_error[i,j] = sequential_mse(reconstruction, targets[:,:,j])
    
    #Apply transformation   
    if transform == 'fisher':
        reconstruction_error = np.arctanh(reconstruction_error)

    #Get Directionality measure as well
    directionality = reconstruction_error - reconstruction_error.T

    if return_pval:
        surrogates = np.zeros((N,n_surrogates,tDelay))*np.nan
        if parallel:
            with Pool(MAX_PROCESSES) as p:
                surrogates[mask_u_idx,:,:] = np.array(list(p.map(partial(twin_surrogates,N=n_surrogates), delay_vectors[:,:,mask_u_idx].transpose([2,0,1]))))
                results = p.map(partial(connectivity,test_ratio=test_ratio,delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask,transform=transform),surrogates.transpose([1,2,0]))

                # connectivity_surr = np.array(list(p.map(partial(connectivity,test_ratio=test_ratio,delay=delay,
                #              dim=dim,n_neighbors=n_neighbors,method=method,mask=mask,transform=transform), surrogates.transpose([1,2,0])))).transpose([1,2,0])
        else:    
            surrogates[mask_u_idx,:,:] = np.array(list(map(lambda x: twin_surrogates(x,n_surrogates), delay_vectors[:,:,mask_u_idx].transpose([2,0,1]))))

            results = map(lambda x: connectivity(x,test_ratio=test_ratio,delay=delay,dim=dim,n_neighbors=n_neighbors,method=method,mask=mask, transform=transform), surrogates.transpose([1,2,0]))
            # connectivity_surr = np.array(list(map(lambda x: connectivity(x,test_ratio=test_ratio,delay=delay,
            #              dim=dim,n_neighbors=n_neighbors,method=method,mask=mask, transform=transform), surrogates.transpose([1,2,0])))).transpose([1,2,0])
            
        #Get surrogate results
        connectivity_surr = np.array([r[0] for r in results]).transpose([1,2,0])
        directionality_surr = np.array([r[1] for r in results]).transpose([1,2,0])

        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_FCF = 1-2*np.abs(np.array([[stats.percentileofscore(connectivity_surr[i,j,:],reconstruction_error[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

        #Calculate the signifance of the unshuffled FCF results given the surrogate distribution
        pval_dir = 1-2*np.abs(np.array([[stats.percentileofscore(directionality_surr[i,j,:],directionality[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

        # #One way to get the significance is to calculate the z-score of the FCF relative to the surrogate distribution and pass it through the surival function; this gets similar signifant elements
        # pval_FCF = stats.norm.sf((reconstruction_error - np.mean(connectivity_surr,axis=-1))/np.std(connectivity_surr,axis=-1))

                    
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error,'pval_FCF':pval_FCF,'directionality':directionality,'pval_dir':pval_dir,'surrogates':surrogates,'connectivity_surr':connectivity_surr,'n_surrogates':n_surrogates,
                                 'test_ratio':test_ratio,'delay':delay,'dim':dim,'n_neighbors':n_neighbors,
                                 'method':method,'mask':mask,'transform':transform})

        return reconstruction_error, pval_FCF, directionality, pval_dir
    else:
        
        if save_data:
            savemat(file+'.mat',{'fcf':reconstruction_error,'directionality':directionality,'test_ratio':test_ratio,'delay':delay,'dim':dim,'n_neighbors':n_neighbors,
                                 'method':method,'mask':mask,'transform':transform})
            
        return reconstruction_error, directionality


def build_nn(X,train_indices,test_indices,test_ratio=.02,n_neighbors=4):
    """Build nearest neighbour data structures for multiple delay vectors
    
    Args:
        X (numpy.ndarray): 3D (TxDxN) numpy array of delay vectors for
            multiple signals
        train_indices (array): indices used for the inference of the CCM
            mapping
        test_indices (array): indices used for applying the inferred CCM
            mapping and further reconstruction
    Returns:
        array: Nearest neighbor data structures (see the documentation of 
                 NearestNeighbors.kneighbors)
    
    """
    nns = []    
    for i in range(X.shape[2]):
        nbrs = NearestNeighbors(n_neighbors, algorithm='ball_tree').fit(X[train_indices,:,i])
        distances, indices = nbrs.kneighbors(X[test_indices,:,i])
        weights = np.exp(-distances)
        weights = weights/(weights.sum(axis=1)[:,np.newaxis])
        nns.append((weights,indices))
    return nns


def estimate_dimension(X, tau, method='fnn'):
    """Estimate the embedding dimension from the data
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate
            the embedding dimension
        tau (integer): Taken time delay 
        method (string): Method for estimating the embedding dimension, choose
            from ('fnn', hilbert)
        
    Returns:
        integer: Estimated embedding dimension
    
    """
    
    # TODO: Implement correlation dimension and box counting dimension methods
    
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


def estimate_timelag(X,method='autocorr'):
    """Estimate the embedding time lag from the data
    
    Args:
        X (numpy.ndarray): (TxN) multivariate signal for which we want to estimate
            the embedding time lag
        method (string): Method for estimating the embedding time lag tau, choose
            from ('autocorr', 'mutinf')
        
    Returns:
        integer: Estimated embedding time lag
    
    TODO: mutinf section does not work

    """
    
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
    else:
        raise Exception('Method {} not implemented'.format(method))
    
    return tau


def twin_surrogates(X,N):
    """Create twin surrogates for significance evaluation and hypothesis testing
    
    Args:
        X (numpy.ndarray): (NxT) multivariate signal for which we want to estimate
            the embedding dimension
        N (integer): Number of surrogate datasets to be created
        
    Returns:
        numpy.ndarray: Generated twin surrogate dataset
    
    """
    
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
