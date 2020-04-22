# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:54:00 2020

@author: Amin
"""

import Simulator
import numpy as np
import matplotlib.pyplot as plt
import MultivariateGrangerCausality as mgc

parameters = {}

parameters['N']     = 1000    # number of neurons
parameters['tau']   = 0.0001    # time constant (seconds)
parameters['T']     = 2       # full duration (seconds)
parameters['g']     = 2.5     # controls the variance of connections
parameters['Rmax']  = 2       # R_max
parameters['R0']    = 0.1*parameters['Rmax'] # R_0
parameters['inp_t'] = 'step'   # type of input
parameters['inp_h'] = 2      # I_{1/2}
parameters['inp']   = 1.5     # amount of input step current
parameters['fs']    = 1000     # sampling frequency
parameters['spon']  = 2       # (seconds)

r, t, I = Simulator.larry_model(parameters)
#plt.plot(t,r)

r = r[100:,:]
t = t[100:]

print(t.shape)
print(r.shape)
plt.plot(t,r[:,0:10])
plt.show()

import DelayEmbedding as DE

DE_PARAMS = {}
DE_PARAMS['tau'] = 10
DE_PARAMS['D'] = 5


delay_vectors = np.array(list(map(lambda x: DE.create_delay_vector(x,DE_PARAMS['D'],DE_PARAMS['tau']), r.T)))
print(delay_vectors.shape)

DE_PARAMS['n_neighbours'] = 3
DE_PARAMS['T_lib'] = len(t) - 5000


for i in range(r.shape[1]):
    for j in range(r.shape[1]):
        recon = DE.reconstruct(delay_vectors[i,DE_PARAMS['T_lib']:,:], \
                               delay_vectors[i,:DE_PARAMS['T_lib'],:], \
                               delay_vectors[j,:DE_PARAMS['T_lib'],:], DE_PARAMS['n_neighbours'])
        
        print(DE.sequentialCorr(recon, delay_vectors[j,DE_PARAMS['T_lib']:,:]))

np.corrcoef(r[:,0], r[:,1])[0,1]





G = mgc.tsdata_to_autocov(r[0:1000,0:100], 5)
AF, SIG = mgc.autocov_to_var(G)

mgc.autocov_to_mvgc(G, np.array([0, 1, 2]), np.array([4, 5, 6]))
