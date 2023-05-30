#Base
import os
import ray
from ray.actor import ActorHandle
import numpy as np
import scipy.stats as st

#Model
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

#User
from delay_embedding import evaluation as E
from delay_embedding import helpers as H
from delay_embedding import surrogate as S

@ray.remote
class FCF:

    def __init__(self, X: np.array=None, pair: tuple=None, data_fpath: str=None, mask: np.array=None,
                 test_ratio=.1, delay: int=10, dim: int=3, n_neighbors: int=4,
                 transform: str='fisher', rand_proj: bool=False, save_reconstructions: bool=False):
        
        """Initialize a new class for performing functional connectivity analyses

        Args:
            X (numpy.ndarray): Multivariate signal to compute functional connectivity from (TxN), columns are the time series for different chanenls/neurons/pixels
            data_fpath (str) : String to npy file containing array
            delay (integer): Delay embedding time delay 
            dim (integer): Delay embedding dimensionality
            n_neighbors (integer): for k-nearest neigbors data structure
            transform (string): Transformation applied to the inferred functional connectivity, choose from ('fisher','identity')
            rand_proj (bool): Random projection of delay vectors for a more isotropic representation
        """

        if (data_fpath is None) & (X is None):
            raise Exception('No data array or filepath were provided. Try again.')
        
        #Dataset of interest
        self.data_fpath = data_fpath
        if X is None:
            X =  np.load(self.data_fpath)

        self.pair = pair
        #If pair is not none, then we're only going to look at a pair of nodes within X;
        #This will be useful when we want to parallelize over surrogates + kfolds, 
        #whilst simultaneously optimizing for takens/tau for each pair & keeping memory at a minimum
        self.X = np.squeeze(X[:,pair])
        self.mask = mask
        
        #Model parameters
        self.delay = delay
        self.dim = dim
        self.n_neighbors = n_neighbors
        self.transform = transform
        self.rand_proj = rand_proj
        self.test_ratio = test_ratio

        #Create cache for saving arrays
        self._trace_cache = {}
        self.save_reconstructions = save_reconstructions
        self.reconstructions = {}

    def get_data_dimensions(self) -> tuple:
        return self.X.shape

    def get_data_array(self) -> np.array:
        return self.X
    
    def get_reconstructions(self) -> dict:
        return self.reconstructions
        
    def get_ccm_parameters(self) -> tuple:
        return (self.delay,self.dim,self.n_neighbors,self.transform)
        
    def get_delay_vectors(self, train_indices: np.array=None, test_indices: np.array=None, test_ratio: float=.025, iFold: int=None, reload: bool=False) -> tuple:
        
        cache_key = ('delay_vectors',iFold)
        # Check if cached
        if (cache_key in self._trace_cache) & (~reload):
            return self._trace_cache[cache_key]

        delay, dim, _, _ = self.get_ccm_parameters()
        T = self.X.shape[0]
        tShift = delay*(dim-1)    #Max time shift
        tDelay = T - tShift       #Length of delay vectors

        #Since no train_test indices were given, we are not doing cross-validation and therefore can create and split the delay vectors normally
        if train_indices is None:

            #How much data are we going to try and reconstruct?
            tTest = np.max([1.0,np.min([np.floor(test_ratio*tDelay),tDelay-tShift-1.0])]).astype(int)
            
            #Get indices for training and test datasets
            iStart = tDelay - tTest; iEnd = tDelay
            test_indices = np.arange(iStart,iEnd)
            train_indices = np.arange(0,iStart-tShift)

            #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
            delay_vectors = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X.T)),2)

            #Split delay vectors into those used for the inference of the CCM mapping and those used for testing the reconstruction
            cue_dvs = delay_vectors[train_indices,:,:]
            target_dvs = delay_vectors[test_indices,:,:]
            
            
        #Since train and test indices were given, we must create the delay_vectors a bit differently
        else:
            #Reconstruct the attractor manifold for each node in the delay coordinate state space
            #and split train and test delay vectors
            #Get index where training set is split in 2
            bb = np.where(np.diff(train_indices) != 1)[0]
            if len(bb) == 1:
                train1 = train_indices[0:bb[0]+1]
                train2  = train_indices[bb[0]+1:]
                
                #Create delay vectors for each block of training set and combine
                dv1 = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X[train1].T)),2)
                dv2 = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X[train2].T)),2)
                cue_dvs = np.concatenate((dv1,dv2),0)

                #Create delay vector for test set
                target_dvs = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X[test_indices].T)),2)

            else:
                #For first and last test set, the training set is 1 continuous block
                cue_dvs = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X[train_indices].T)),2)
                target_dvs = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,dim)[:,:,np.newaxis], self.X[test_indices].T)),2)

        # Save in cache
        self._trace_cache[cache_key] = (cue_dvs, target_dvs)

        return (cue_dvs, target_dvs)
    
    def project_delay_vectors(self, cue_dvs: np.array, target_dvs: np.array) -> tuple:
    
        T, D, N = cue_dvs.shape
        proj_cue_dvs = np.zeros(cue_dvs.shape)
        proj_target_dvs = np.zeros(target_dvs.shape)

        projection_matrices = []
        for i in range(N):
            P =  np.random.rand(D,self.dim)
            projection_matrices.append(P)
            proj_cue_dvs[:,:,i] = cue_dvs[:,:,i] @ P
            proj_target_dvs[:,:,i] = target_dvs[:,:,i] @ P

        #Save projection matrices
        self.proj_matrices = np.array(projection_matrices)

        return (proj_cue_dvs, proj_target_dvs)
    
    def build_nn(self, cue_dvs: np.array, target_dvs: np.array) -> list:

        # Build nearest neighbour data structures for multiple delay vectors
        # nns is a list of tuples: the first being the weights used for the forecasting technique, 
        # and the second element the elements corresponding to the training delay vector

        nns = []    
        N = cue_dvs.shape[-1]
        for i in range(N):
            nbrs = NearestNeighbors(n_neighbors=self.n_neighbors, algorithm='ball_tree').fit(cue_dvs[:,:,i])
            distances, indices = nbrs.kneighbors(target_dvs[:,:,i])
            weights = np.exp(-distances)
            weights = weights/(weights.sum(axis=1)[:,np.newaxis])
            nns.append((weights,indices))
        return nns

    def calculate_connectivity(self, cue_dvs: np.array=None, target_dvs: np.array=None, pba: ActorHandle=None):

        #If cue_dvs is not provided, get dvs based on test_ratio
        if cue_dvs is None:
            (cue_dvs, target_dvs) = self.get_delay_vectors(test_ratio=self.test_ratio,reload=True)

        if self.rand_proj:
            (cue_dvs, target_dvs) = self.project_delay_vectors(cue_dvs, target_dvs)
    
        T, N = self.X.shape
        delay, dim, n_neighbors, transform = self.get_ccm_parameters()
        mask = self.mask

        # Calculate reconstruction error only for elements we want to compute
        if mask is None: 
            mask = np.zeros((N,N)).astype(bool)
            mask[np.diag_indices(N)] = True #Don't reconstruct diagonal
        mask_idx = np.where(~mask)
        mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
        
        # Build nearest neighbour data structures for multiple delay vectors
        nns_ = self.build_nn(cue_dvs, target_dvs)
        nns = [[]]*N
        for i,idx in enumerate(mask_u_idx):
            nns[idx] = nns_[i]
        
        # Loop over each pair and calculate the topological embeddedness for signal i by another signal j
        # This will be quantified based on the correlation coefficient between the true and forecast signals
        reconstruction_error = np.zeros((N,N))*np.nan
        for i, j in zip(*mask_idx):
            # Use k-nearest neigbors forecasting technique to estimate the forecast signal
            reconstruction = np.array([nns[i][0][idx,:]@cue_dvs[nns[i][1][idx,:],:,j] for idx in range(target_dvs.shape[0])])
            reconstruction_error[i,j] = E.sequential_correlation(reconstruction, target_dvs[:,:,j])

            if self.save_reconstructions:
                self.reconstructions[(i,j)] = reconstruction
        
        # Apply transformation   
        if transform == 'fisher': fcf = np.arctanh(reconstruction_error)
        if transform == 'identity': fcf = reconstruction_error
        
        #Update ray progress bar
        if pba is not None:
            pba.update.remote(1)

        return fcf
    

    def cross_validate(self, nKfold: int=10, pba: ActorHandle=None, surrogate: bool=False) -> np.array:
    
        T, N = self.X.shape
        FCF_kfold = np.zeros((nKfold,N,N))*np.nan

        k_fold = KFold(n_splits=nKfold)
        for iK, (train_indices, test_indices) in enumerate(k_fold.split(self.X)):
            (cue_dvs, target_dvs) = self.get_delay_vectors(train_indices=train_indices, test_indices=test_indices, iFold=iK)

            if surrogate:
                #Create surrogate datasets from training data
                # refs = [S.twin_surrogates_remote.remote(cue_dvs[:,:,i],N=1) for i in range(cue_dvs.shape[2])]
                # surrogates = np.squeeze(ray.get(refs)).T
                surrogate_X = np.squeeze([S.twin_surrogates(cue_dvs[:,:,i],N=1) for i in range(cue_dvs.shape[2])]).T
                surrogate_dvs = np.concatenate(list(map(lambda x: H.create_delay_vector(x,self.delay,self.dim)[:,:,np.newaxis], surrogate_X.T)),2)
                FCF_kfold[iK] = self.calculate_connectivity(cue_dvs=surrogate_dvs, target_dvs=target_dvs,pba=pba)
            else:
                FCF_kfold[iK] = self.calculate_connectivity(cue_dvs=cue_dvs, target_dvs=target_dvs,pba=pba)

        return FCF_kfold

##===== Helper functions that use FCF class as input =====##
# Note: we're only going to run this function once we have optimized takens + tau for each pair. 
def surrogate_connectivity(X: np.array, takens: np.array, tau: np.array, mask: np.array=None,
                           n_neighbors: int=4,transform: str='fisher', rand_proj: bool=False,
                           nKfold: int=10, n_surrogates: int=100, pba: ActorHandle=None) -> np.array:
    
    #Put data array into ray object store memory
    X_id = ray.put(X)
    T, N = X.shape

    # Calculate reconstruction error only for elements we want to compute
    if mask is None: 
        mask = np.zeros((N,N)).astype(bool)
        mask[np.diag_indices(N)] = True #Don't reconstruct diagonal
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))

    #Preallocate
    FCF_surrogate = np.zeros((n_surrogates,nKfold,N,N))*np.nan

    #Loop over all unique combinations of the taken's dimension and delay time
    for dim in np.unique(takens):
        for delay in np.unique(tau):
            
            mask2 = (takens == dim) & (tau == delay) & ~mask
            if (np.sum(mask2) == 0) | (dim == 0):
                continue

            print(f'Calculating cross-validated surrogate FCF distribution for a takens dimension {dim} and a time delay {delay}; {np.sum(mask2)} elements')

            #Create progress bar
            pb = H.ProgressBar(n_surrogates*nKfold)
            actor = pb.actor

            ## Create n_surrogate class instances of FCF, passing in the ID to the original data array
            obj_list = [FCF.remote(X=X_id, mask=mask2, delay=delay,dim=dim,n_neighbors=n_neighbors,transform=transform) for _ in range(n_surrogates)]

            #Start a series of remote Ray tasks 
            processes = [fcf.cross_validate.remote(nKfold=nKfold ,pba=actor, surrogate=True) for fcf in obj_list]
            
            #And then print progress bar until done
            pb.print_until_done()

            #Initiate parallel processing
            tmp = np.array(ray.get(processes))
            
            #Save   
            FCF_surrogate[:,:,mask2] = tmp[:,:,mask2]
        
    return FCF_surrogate



# ##===== Helper functions =====##
# def surrogate_connectivity(X: np.array, takens: np.array, tau: np.array, mask: np.array=None,
#                            n_neighbors: int=4,transform: str='fisher', rand_proj: bool=False,
#                            nKfold: int=10, n_surrogates: int=100, pba: ActorHandle=None) -> np.array:
    
#     #Get the maximum takens dimension and time delay to create delay vectors with
#     max_takens = np.max(takens)
#     max_delay = np.max(tau)

#     #Put data array into ray object store memory
#     X_id = ray.put(X)
#     T, N = X.shape

#     # Calculate reconstruction error only for elements we want to compute
#     if mask is None: mask = np.zeros((N,N)).astype(bool)
#     mask[np.diag_indices(N)] = True #Don't reconstruct diagonal
#     mask_idx = np.where(~mask)
#     mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))

#     #Create progress bar
#     pb = H.ProgressBar(n_surrogates)
#     actor = pb.actor

#     #Since delay vectors are created based on the time delay and embedding dimension, we'll loop over the delay
#     for delay in np.unique(tau):
#         #Reconstruct the attractor manifold for each node in the delay coordinate state space; size [tDelay x dim x N]
#         delay_vectors = np.concatenate(list(map(lambda x: H.create_delay_vector(x,delay,max_takens)[:,:,np.newaxis], X.T)),2)

#         #Create surrogate datasets
#         ref = []
#         for iN in range(N):
#             #Take the mode across the row for surrogate takens dimension
#             dim = np.mode(takens[iN])
#             ref.append(S.twin_surrogates.remote(delay_vectors[:,:dim,iN],N=n_surrogates))
#     refs = [S.twin_surrogates.remote(delay_vectors[:,:,i],N=n_surrogates) for i in range(delay_vectors.shape[2])]
#     surrogates = np.array(ray.get(refs),dtype=object)
#     surrogates = np.transpose(surrogates,[1,2,0])

#     ## Create n_surrogate class instances of FCF
#     obj_list = [FCF.remote(X=surr, delay=delay,dim=dim,n_neighbors=n_neighbors,transform=transform) for surr in surrogates]

#     #Start a series of remote Ray tasks 
#     processes = [fcf.calculate_connectivity.remote(pba=actor) for fcf in obj_list]

#     #And then print progress bar until done
#     pb.print_until_done()

#     #Initiate parallel processing
#     fcf_surrogates = np.array(ray.get(processes))
    
#     return fcf_surrogates



# ##===== Helper functions that use FCF class as input =====##
# # Note: we're only going to run this function once we have optimized takens + tau for each pair. 
# def surrogate_connectivity(X: np.array, takens: np.array, tau: np.array, mask: np.array=None,
#                            n_neighbors: int=4,transform: str='fisher', rand_proj: bool=False,
#                            nKfold: int=10, n_surrogates: int=100, pba: ActorHandle=None) -> np.array:
    
#     #Put data array into ray object store memory
#     X_id = ray.put(X)
#     T, N = X.shape

#     if mask is None: 
#         mask = np.zeros((N,N)).astype(bool)
#         mask[np.diag_indices(N)] = True #Don't reconstruct diagonal
#     mask_idx = np.where(~mask)
#     mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
#     #Preallocate
#     FCF_surrogate = np.zeros((n_surrogates,nKfold,2,N,N))*np.nan
    
#     # sub_mask = np.array([[0,0],[0,1]])
#     #Loop over each pair and calculate FCF for it's optimal takens/tau pair
#     for iN, jN in zip(*mask_idx):
#         dim, delay = takens[iN,jN],tau[iN,jN]
#         if (iN == jN) | (dim == 0):
#             continue

#         #Create progress bar
#         pb = H.ProgressBar(n_surrogates*nKfold)
#         actor = pb.actor

#         ## Create n_surrogate class instances of FCF, passing in the ID to the original data array
#         obj_list = [FCF.remote(X=X_id, pair=(iN,jN), delay=delay,dim=dim,n_neighbors=n_neighbors,transform=transform) for _ in range(n_surrogates)]

#         #Start a series of remote Ray tasks 
#         processes = [fcf.cross_validate.remote(nKfold=nKfold ,pba=actor, surrogate=True) for fcf in obj_list]
        
#         #And then print progress bar until done
#         pb.print_until_done()

#         #Initiate parallel processing
#         tmp = np.array(ray.get(processes))

#         #Save
#         FCF_surrogate[:,:,0,iN,jN] = tmp[:,:,1,0]
#         FCF_surrogate[:,:,1,jN,iN] = tmp[:,:,0,1]

#     return FCF_surrogate

def calculate_significance(fcf: np.array, fcf_surrogates: np.array) -> np.array:
    N = fcf.shape[0]

    # Calculate the signifance of the unshuffled FCF results given the surrogate distribution
    pval = 1-2*np.abs(np.array([[st.percentileofscore(fcf_surrogates[:,i,j],fcf[i,j],kind='strict') for j in range(N)] for i in range(N)])/100 - .5)

    return pval

