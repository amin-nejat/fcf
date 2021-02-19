# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:54:00 2020

@author: Amin
"""

from DelayEmbedding import DelayEmbedding as DE
from Causality import responseAnalysis as RA
from Simulation.Simulator import Simulator
import visualizations as V
import numpy as np

# %% Resting State

parameters = {}
parameters['T'] = 1000
parameters['alpha'] = .2
parameters['beta'] = .2
parameters['gamma'] = 5.7
parameters['bernoulli_p'] = .8
parameters['g_i'] = .1
parameters['g_r'] = 3. # 2 for full downstreamness
parameters['lambda'] = 1.
parameters['N'] = 100


t,y,J = Simulator.rossler_downstream(parameters)

recorded = np.arange(10)
V.visualize_signals(t,[y[:,recorded].T],['Observations'])
V.visualize_matrix(J[recorded,:][:,recorded],cmap='coolwarm',titlestr='Connectome')

# %% Compute functional causal flow (FCF)

fcf = DE.connectivity(y[:,recorded],test_ratio=.1,delay=1,dim=5,n_neighbors=3,return_pval=False)
V.visualize_matrix(fcf,titlestr='FCF',cmap='cool')

# %% Stimulation

N = parameters['N']
I,t_stim_prot,_,stimulated = Simulator.stimulation_protocol([(i,i+1) for i in recorded],time_st=min(t),
      time_en=max(t),N=N,n_record=1,stim_d=.2,rest_d=1,feasible=np.ones(N).astype(bool),
      amplitude=10*np.ones(N),repetition=1,fraction_stim=1,visualize=True)

parameters_stim = parameters.copy()
parameters_stim['J'] = J
parameters_stim['I'] = I
parameters_stim['t'] = t_stim_prot
parameters_stim['I_J'] = np.eye(parameters_stim['N'])

t_stim,y_stim,_ = Simulator.rossler_downstream(parameters_stim)

V.visualize_signals(t_stim,[y_stim[:,recorded].T],['Observations'],stim=I[:,recorded],stim_t=t_stim_prot)

# %% Interventional Connectivity

stim_s = np.where(np.diff(I[:,recorded].T,axis=1) > 0)
stim_e = np.where(np.diff(I[:,recorded].T,axis=1) < 0)

stim_e = stim_s

stim_d = [t_stim_prot[stim_e[1][i]] - t_stim_prot[stim_s[1][i]] for i in range(len(stim_s[1]))]
stim = [(stim_s[0][i], t_stim_prot[stim_s[1][i]], t_stim_prot[stim_e[1][i]]) for i in range(len(stim_s[1]))]

output = RA.interventional_connectivity(y_stim[:,recorded].T,stim,t=t_stim,
                bin_size=50,skip_pre=.0,skip_pst=.0,pval_threshold=1,
                methods=['aggr_ks'])

V.visualize_matrix(output['aggr_ks'].T,titlestr='Interventional Connectivity',cmap='copper')