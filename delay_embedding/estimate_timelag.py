# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 17:15:48 2022

@author: Amin
"""
import numpy as np

# %%
def timelag_autocorr(X):
    x = np.arange(len(X)).T
    FM = np.ones((len(x),4))
    for pn in range(1,4):
        CX = x**pn
        FM[:,pn] = (CX-CX.mean())/CX.std()
    
    csig = X-FM@(np.linalg.pinv(FM)@X)
    acorr = np.real(np.fft.ifft(np.abs(np.fft.fft(csig))**2).min(1))
    tau = np.where(np.logical_and(acorr[:-1]>=0, acorr[1:]<0))[0][0]
    
    return tau

# %%
def timelag_mutinfo(X):
    # TODO: mutinf section does not work
    L = len(X)
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
# %%
def estimate_timelag(X,method='autocorr'):
    '''Estimate the embedding time lag from the data
    
    Args:
        X (numpy.ndarray): (TxN) multivariate signal for which we want to estimate
            the embedding time lag
        method (string): Method for estimating the embedding time lag tau, choose
            from ('autocorr', 'mutinf')
    Returns:
        integer: Estimated embedding time lag
    '''
    if method == 'autocorr':
        return timelag_autocorr(X)
    if method == 'mutinf':
        return timelag_mutinfo(X)
    


# %%
def autocorrelation_func(x):
    '''Autocorrelation function
        http://stackoverflow.com/q/14297012/190597
        
        Args:
            x (np.ndarray): signal
    '''
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    assert np.allclose(r, np.array([(x[:n-k]*x[-(n-k):]).sum() for k in range(n)]))
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

# %%
def find_correlation_time(x,dt,nlags=100):
    '''Autocorrelation time of a multivariate signal
    
    Args:
        x (np.ndarray): is a time series on a regular time grid
        dt (float): is the time interval between consecutive time points
    
    Returns:
        float: Autocorrelation time
    '''
    C = autocorrelation_func(x)
    if  len(np.where(C<0)[0])>0:
        C = C[:(np.where(C<0)[0][0])]
    return (dt*np.sum(C)/C[0])
