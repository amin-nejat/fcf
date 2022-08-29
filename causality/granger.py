# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: Amin
"""

from statsmodels.tsa.stattools import grangercausalitytests

import numpy as np
import scipy as sp

import os
# %%
'''
python MVGC rewritten from Matlab MVGC toolbox

refs:

http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/gc/autocov_to_mvgc.m
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/core/autocov_to_var.m
https://github.com/benpolletta/MBP-Code/blob/master/mvgc_v1.0/core/tsdata_to_autocov.m
'''

# %%
def detrend_mean(x, axis=None):
    # https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/mlab.py
    x = np.asarray(x)

    if axis is not None and axis+1 > x.ndim:
        raise ValueError('axis(=%s) out of bounds' % axis)

    # short-circuit 0-D array.
    if not x.ndim:
        return np.array(0., dtype=x.dtype)

    # short-circuit simple operations
    if axis == 0 or axis is None or x.ndim <= 1:
        return x - x.mean(axis)

    ind = [slice(None)] * x.ndim
    ind[axis] = np.newaxis
    return x - x.mean(axis)[ind]


# %%
def tsdata_to_autocov(X, q):
    '''Calculate sample autocovariance sequence from time series data
    
    Args:
        X: multi-trial time series data
        q: number of lags
    
    Returns:
        G: sample autocovariance sequence
    '''
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=2)
        [n, m, N] = np.shape(X)
    else:
        [n, m, N] = np.shape(X)
    X = detrend_mean(X, axis=1)
    G = np.zeros((n, n, (q+1)))

    for k in range(q+1):
        M = N * (m-k)
        Xk = np.asmatrix(X[:, k:m, :].reshape((n, M)))
        Xm = np.asmatrix(X[:, 0:m-k, :].reshape((n, M)).conj().T)
        G[:, :, k] = Xk * Xm / M - 1.0
        
    # G.shape = (n, n, q+1)
    return G

# %%
def autocov_to_var(G):
    '''Calculate VAR parameters from autocovariance sequence

    Args:
        G: autocovariance sequence
        
    Returns:
        [A, SIG]: A - VAR coefficients matrix, SIG - residuals covariance matrix
    '''
    [n, _, q1] = G.shape
    q = q1 - 1
    qn = q * n
    G0 = np.asmatrix(G[:, :, 0])                                            # covariance
    GF = np.asmatrix(G[:, :, 1:].reshape((n, qn),order='F').T)                        # forward autoconv seq
    GB = np.asmatrix(G[:, :, 1:].transpose(0, 2, 1).reshape((qn, n), order='F'))         # backward autoconv seq

    AF = np.asmatrix(np.zeros((n, qn)))                                     # forward  coefficients
    AB = np.asmatrix(np.zeros((n, qn)))                                     # backward coefficients

    # initialise recursion
    k = 1                                                                   # model order
    r = q - k
    kf = np.arange(0, k*n)                                                  # forward indices
    kb = np.arange(r*n, qn)                                                 # backward indices

    G0I = G0.I

    AF[:, kf] = GB[kb, :] * G0I
    AB[:, kb] = GF[kf, :] * G0I

    for k in np.arange(2, q+1):
        GBx = GB[(r-1)*n: r*n, :]
        DF = GBx - AF[:, kf] * GB[kb, :]
        VB = G0 - AB[:, kb] * GB[kb, :]
        AAF = DF * VB.I

        GFx = GF[(k-1)*n: k*n, :]
        DB = GFx - AB[:, kb] * GF[kf, :]
        VF = G0 - AF[:, kf] * GF[kf, :]
        AAB = DB * VF.I

        AFPREV = AF[:, kf]
        ABPREV = AB[:, kb]
        r = q - k

        kf = np.arange(0, k*n)
        kb = np.arange(r*n, qn)
        
        AF[:, kf] = np.hstack([AFPREV - AAF * ABPREV, AAF])
        AB[:, kb] = np.hstack([AAB, ABPREV - AAB * AFPREV])

    SIG = G0 - AF * GF
    AF = np.asarray(AF).reshape((n, n, q), order='F')
    return [AF, SIG]

# %%
def autocov_to_mvgc(G, x, y, epsilon=1e-6):
    '''Calculate conditional time-domain MVGC (multivariate Granger causality)

    from the variable |Y| (specified by the vector of indices |y|)
    to the variable |X| (specified by the vector of indices |x|),
    conditional on all other variables |Z| represented in |G|, for
    a stationary VAR process with autocovariance sequence |G|.
    See ref. [1] for details.
    [1] L. Barnett and A. K. Seth, <http://www.sciencedirect.com/science/article/pii/S0165027013003701
    The MVGC Multivariate Granger Causality Toolbox: A New Approach to Granger-causalInference>,
    _J. Neurosci. Methods_ 223, 2014 [ <matlab:open('mvgc_preprint.pdf') preprint> ].

    Ref:
    http://erramuzpe.github.io/C-PAC/blog/2015/06/10/multivariate-granger-causality-in-python-for-fmri-timeseries-analysis/
    https://github.com/benpolletta/MBP-Code/blob/ced60ba9fdcea1efd185dc331023e788b9c48eb6/mvgc_v1.0/gc/autocov_to_mvgc.m

    Args:
        G:   autocovariance sequence
        x:   vector of indices of target (causee) multi-variable
        y:   vector of indices of source (causal) multi-variable
    
    Returns:
        F: Granger causality
    '''
    
    if x == y:
        return np.nan
    
    n = G.shape[0]
    z = np.arange(n)
    z = np.delete(z, [np.hstack((x, y))])
    # indices of other variables (to condition out)
    xz = np.hstack((x, z))
    xzy = np.hstack((xz, y))
    F = 0
    
    # full regression
    [AF, SIG] = autocov_to_var(G[xzy,:][:,xzy])
    SIG += epsilon*np.eye(len(SIG)) # Well condition
    # reduced regression
    [AF, SIGR] = autocov_to_var(G[xz,:][:,xz])
    SIGR += epsilon*np.eye(len(SIGR)) # Well condition
    
    F = np.log(np.linalg.det(SIGR[np.arange(len(x)),:][:,np.arange(len(x))])) - \
        np.log(np.linalg.det(SIG [np.arange(len(x)),:][:,np.arange(len(x))]))
    return F


# %%
def multivariate_gc(data,maxlag=2,mask=None,save=False,load=False,file=None):
    '''Multivariate Granger Causality runner for all pairs in a multivariate signal

    Args:
        data (np.array): Multivariate signal (N x T).
        maxlag (integer, optional): Maximum Lag for Granger Causality analysis. The default is 2.

    Returns:
        gc (np.array): MVGC matrix (N x N).

    '''
    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']
    
    if mask is None:
        mask = np.zeros((len(data),len(data))).astype(bool)
    
    mask_idx = np.where(~mask)
    mask_u_idx = np.unique(np.concatenate((mask_idx[0],mask_idx[1])))
    
    G = tsdata_to_autocov(data[mask_u_idx,:], maxlag)
    gc = np.zeros(mask.shape)*np.nan
    gc[mask_idx] = np.array([autocov_to_mvgc(G,[i],[j]) for i,j in zip(*mask_idx)])
    
    p_values = 1-sp.stats.chi2.cdf(gc*(data.shape[1]-maxlag), data.shape[1]-maxlag)
        
    if save: np.save(file,{'cnn':gc, 'pvalue':p_values, 'autocov':G})
    return gc,p_values

# %%
def univariate_gc(data,test='ssr_chi2test',mask=None,maxlag=2,save=False,load=False,file=None):    
    '''Check Granger Causality of all possible combinations of the Time series.
    The rows are the response variable, columns are predictors. The values in the table 
    are the P-Values. P-Values lesser than the significance level (0.05), implies 
    the Null Hypothesis that the coefficients of the corresponding past values is 
    zero, that is, the X does not cause Y can be rejected.

        data      : pandas dataframe containing the time series variables
        variables : list containing names of the time series variables.
    '''
    if load and os.path.exists(file):
        result = np.load(file,allow_pickle=True).item()
        return result['cnn'],result['pvalue']
    
    if mask is None:
        mask = np.zeros((len(data),len(data))).astype(bool)
    mask_idx = np.where(~mask)
    gc = np.zeros(mask.shape)*np.nan
    p_values = np.zeros(mask.shape)*np.nan
    
    for r,c in zip(*mask_idx):
        test_result = grangercausalitytests(data[[r,c]].T, maxlag=maxlag, verbose=False)
        gc[r,c] = np.log(max([round(test_result[i+1][0][test][0],4) for i in range(maxlag)]))
        p_values[r,c] = min([round(test_result[i+1][0][test][1],4) for i in range(maxlag)])
        
    if save: np.save(file,{'cnn':gc,'pvalue':p_values})
    
    return gc,p_values
