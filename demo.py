# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:54:00 2020

@author: Amin
"""
from causality import interventional as intcnn
from causality import helpers as inth
from delay_embedding import ccm

import simulator.networks as net
import visualizations as V

import numpy as np

# %%
T = 100
dt = .05

# %% Resting State
parameters = {}
# parameters['T'] = 1000 # Duration of stimulation
parameters['alpha'] = .2 # Parameter for rossler
parameters['beta'] = .2 # Parameter for rossler
parameters['gamma'] = 5.7 # Parameter for rossler
parameters['bernoulli_p'] = .8 # Downstream connectivity probability
parameters['g_i'] = .1 # Input connectivity strength
parameters['g_r'] = 3. # Recurrent connectivitys strength
parameters['lambda'] = 1. # Parameter for recurrent dynamics
parameters['N'] = 100 # Number of downstream neurons


network = net.RosslerDownstream(parameters['N'],parameters,discrete=True)
t,y = network.run(T,x0=np.random.randn(1,parameters['N']),dt=dt)
J = network.pm['J']


y = y[:,0]
recorded = np.arange(10)
V.visualize_signals(t,[y[:,recorded].T],['Observations'])
V.visualize_matrix(J[recorded,:][:,recorded],cmap='coolwarm',titlestr='Connectome')

# %% Compute functional causal flow (FCF)
fcf,pval,surrogates = ccm.connectivity(y[:,recorded],test_ratio=.1,delay=1,dim=5,n_neighbors=3,return_pval=True)
V.visualize_matrix(fcf,pval=pval,titlestr='FCF',cmap='cool')

# %% Stimulation
N = parameters['N']
I,t_stim_prot,_,stimulated,u = inth.stimulation_protocol(
            [(i,i+1) for i in recorded],
            time_st=min(t),
            time_en=max(t),
            N=N,
            n_record=1,
            stim_d=.2,
            rest_d=1,
            feasible=np.ones(N).astype(bool),
            amplitude=1*np.ones(N),
            repetition=1,
            fraction_stim=1,
            visualize=True
        )

t_stim,y_stim = network.run(T,x0=np.random.randn(1,parameters['N']),dt=dt,u=u)
y_stim = y_stim[:,0]

V.visualize_signals(t_stim,[y_stim[:,recorded].T],['Observations'],stim=I[:,recorded],stim_t=t_stim_prot)

# %% Interventional Connectivity

stim_s = np.where(np.diff(I[:,recorded].T,axis=1) > 0) # Stimulation start times
stim_e = np.where(np.diff(I[:,recorded].T,axis=1) < 0) # Stimulation end times

stim_e = stim_s

stim_d = [t_stim_prot[stim_e[1][i]] - t_stim_prot[stim_s[1][i]] for i in range(len(stim_s[1]))] # Stimulation duration
stim = [(stim_s[0][i], t_stim_prot[stim_s[1][i]], t_stim_prot[stim_e[1][i]]) for i in range(len(stim_s[1]))] # Stimulation array [(chn,start,end),...]

output = intcnn.interventional_connectivity(
            y_stim[:,recorded].T,
            stim,t=t_stim,
            bin_size=50,
            skip_pre=.0,
            skip_pst=.0,
            pval_threshold=1,
            methods=['aggr_ks']
        )

V.visualize_matrix(output['aggr_ks'].T,titlestr='Interventional Connectivity',cmap='copper')