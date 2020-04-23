# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:54:00 2020

@author: Amin
"""

from Simulator import Simulator
import DelayEmbedding as DE
import numpy as np
import matplotlib.pyplot as plt
import MultivariateGrangerCausality as mgc

parameters = {}

parameters['N']     = 100    # number of neurons
parameters['tau']   = 0.001    # time constant (seconds)
parameters['T']     = 2       # full duration (seconds)
parameters['g']     = 10.0     # controls the variance of connections
parameters['Rmax']  = 2       # R_max
parameters['R0']    = 0.1*parameters['Rmax'] # R_0
parameters['inp_t'] = 'step'   # type of input
parameters['inp_h'] = 2      # I_{1/2}
parameters['inp']   = 0     # amount of input step current
parameters['fs']    = 1000     # sampling frequency
parameters['spon']  = 2       # (seconds)
parameters['connectivity'] = 'Gaussian'
#parameters['conn_prob'] = .99.

r, t, I = Simulator.larry_model(parameters)
#plt.plot(t,r)

r = r[100:,:]
t = t[100:]

print(t.shape)
print(r.shape)
plt.plot(t,r[:,0:100])
plt.show()



#DE_PARAMS = {}
#DE_PARAMS['tau'] = 10
#DE_PARAMS['D'] = 5
#
#
#delay_vectors = np.array(list(map(lambda x: DE.create_delay_vector(x,DE_PARAMS['D'],DE_PARAMS['tau']), r.T)))
#print(delay_vectors.shape)
#
#DE_PARAMS['n_neighbours'] = 3
#DE_PARAMS['T_lib'] = len(t) - 5000
#
#
#recon = DE.reconstruct(delay_vectors[0,DE_PARAMS['T_lib']:,:], \
#                       delay_vectors[0,:DE_PARAMS['T_lib'],:], \
#                       delay_vectors[1,:DE_PARAMS['T_lib'],:], DE_PARAMS['n_neighbours'])
#
#print(DE.sequential_correlation(recon, delay_vectors[1,DE_PARAMS['T_lib']:,:]))





G = mgc.tsdata_to_autocov(r[0:1000,0:100], 5)
AF, SIG = mgc.autocov_to_var(G)
mgc.autocov_to_mvgc(G, np.array([0, 1, 2]), np.array([4, 5, 6]))

import itertools

np.array(list(map(lambda x: DE.reconstruction_accuracy(r[:,x[0]],r[:,x[1]]),list(itertools.combinations(range(r.shape[1]),2)))))


#def causation(pair):
#    print(pair)
#    return DE.reconstruction_accuracy(r[:,pair[0]],r[:,pair[1]])
#    
#import multiprocessing
#pool = multiprocessing.Pool()
#pool.map(causation, 


