# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin
"""

import random
import numpy as np
import networkx as nx 
from functools import partial
from scipy import interpolate
import matplotlib.pyplot as plt
from networkx import convert_matrix
from sklearn.decomposition import PCA
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d as intp
from networkx.algorithms import bipartite
from scipy.linalg import block_diag
from itertools import groupby
from operator import itemgetter
from scipy.spatial.distance import cdist

class Simulator(object):
    
    @staticmethod
    def rossler_network(parameters={'T':1000,'alpha':.2,'beta':.2,'gamma':5.7}):
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
            
        def ode_step(t,x,alpha,beta,gamma,inp):
            dxdt = [-(x[1]+x[2]),
                x[0]+alpha*x[1],
                beta+x[2]*(x[0]-gamma)] + inp(t)
            return dxdt
        
        x0 = np.random.rand(3, 1)
        sol = solve_ivp(partial(ode_step,alpha=parameters['alpha'],beta=parameters['beta'],gamma=parameters['gamma'],inp=inp),
                        [0,parameters['T']],x0.squeeze())
        
        return sol.t, sol.y.T
    
    @staticmethod
    def downstream_network(I,t,parameters):
            
        def ode_step(t, x, J, inp, g_i, g_r, lambd):
            dxdt=-lambd*x + 10*np.tanh(g_r*J@x+g_i*inp(t))
            return dxdt
        
        p = parameters['bernoulli_p']
        N = parameters['N']
        g_r = parameters['g_r']
        g_i = parameters['g_i']
        lambd = parameters['lambda']
        
        noise = np.random.randn(I.shape[0],I.shape[1])*parameters['noise_std']
        
        I_J = Simulator.bipartite_connectivity(I.shape[1],N,p,visualize=False)[I.shape[1]:,:I.shape[1]]
        J = Simulator.normal_connectivity(N,1)
        
        I = I + noise
        
        inp = interpolate.interp1d(t,I_J@I.T,kind='linear',bounds_error=False)
        
        ## Initialize
        x0 = np.random.rand(N, 1)
        sol = solve_ivp(partial(ode_step,J=J,inp=inp,g_i=g_i,g_r=g_r,lambd=lambd),
                        [t.min(),t.max()],x0.squeeze(),t_eval=t)
    
        x  = sol.y
        return t, x.T
        
    @staticmethod
    def larry_network(parameters):
        ## Parameters
        
        def ode_step(t, x, J, I, R0, Rmax, tau):
        
            phi = np.zeros(x.shape)
            phi[x<=0] = R0*np.tanh(x[x<=0]/R0)
            phi[x>0]  = (Rmax-R0)*np.tanh(x[x>0]/(Rmax-R0))
        
            r    = R0 + phi
                
            dxdt = 1/tau*(-x + J@r + I[:, int(np.floor(t-1e-5))])
            
            return dxdt
    
        N     = parameters['N']
        tau   = parameters['tau']*parameters['fs']
        T     = parameters['T']*parameters['fs']
        g     = parameters['g']
        inp   = parameters['inp']
        Rmax  = parameters['Rmax']
        R0    = parameters['R0']
        inp_t = parameters['inp_t']
        spon  = parameters['spon']*parameters['fs']
        
        if 'J' in parameters.keys():
            J  = g*parameters['J'].copy()
        elif parameters['connectivity'] == 'Gaussian':
            J  = Simulator.normal_connectivity(N,g)
        elif parameters['connectivity'] == 'small_world':    
            J = g*Simulator.erdos_renyi_connectivity(N,parameters['conn_prob'])
    
    
    
        ## Input current
        if 'I' in parameters.keys():
            I = parameters['I'].copy()
        elif inp_t == 'zero':
            I = np.zeros((N,T))
        elif inp_t == 'const':
            I = inp*np.ones((N, T))
        elif inp_t == 'step':
            I = np.tile(np.concatenate((np.zeros((1, spon)), inp*np.ones((1, T - spon))),axis=1), (N,1))
            
    
        ## Initialize
        x0      = np.random.rand(N, 1)
    
    
        ## Solving equations
        sol = solve_ivp(partial(ode_step,J=J,I=I,R0=R0,Rmax=Rmax,tau=tau),[0,T],x0.squeeze())
    
    
        x         = sol.y
        t         = sol.t
        phi       = np.zeros(x.shape)
        phi[x<=0] = R0*np.tanh(x[x<=0]/R0)
        phi[x>0]  = (Rmax-R0)*np.tanh(x[x>0]/(Rmax-R0))
    
        r         = (R0 + phi).T
    
        t = t / parameters['fs']
    
        return r, t, I

    
    @staticmethod
    def hansel_network(parameters,dt=.001):
        N = parameters['N']
        
        p = parameters['p'] # connectivity probability
        J = Simulator.erdos_renyi_connectivity(N,p,visualize=False)
        spikes = []
        
        Simulator.last_t = 0
        Simulator.current = np.zeros((J.shape[0]))
        
        def ode_step(t,v,inp,C,theta,g_l,tau1,tau2,v_rest,I_syn_avg):
            [spikes.append((spk,t)) for spk in np.where(v >= theta)[0]]
            Simulator.current *= np.exp(Simulator.last_t-t)
            Simulator.current[v >= theta] += (1/(tau1-tau2))*(np.exp(-t/tau1) - np.exp(-t/tau2))
            
            v[v >= theta] = v_rest
            I_syn = -(I_syn_avg/N)*J@Simulator.current
            dvdt = (1/C)*(-g_l*(v-v_rest)+I_syn+inp(t))
            
            Simulator.last_t = t
            return dvdt
    
        g_l = parameters['g_l'] # mS/cm^2
        I_0 = parameters['I_0'] # micronA/cm^2
        v_rest = parameters['v_rest'] # mV
        theta = parameters['theta'] # mV
        tau1 = parameters['tau1'] # ms
        tau2 = parameters['tau2'] # ms
        I_syn_avg = parameters['I_syn_avg'] # micronA/cm^2
        C = parameters['C'] # micronF/cm^2
        T = parameters['T'] # ms
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[I_0,I_0],kind='nearest',fill_value='extrapolate')
        
         # Initialize
        x0 = np.random.rand(N)*20-40
#        sol = solve_ivp(partial(ode_step,inp=inp,C=C,theta=theta,tau1=tau1,
#                                g_l=g_l,tau2=tau2,v_rest=v_rest,
#                                I_syn_avg=I_syn_avg),[0,T],x0.squeeze())
#        
#        spk = [[]]*N
#        for n in range(N):
#            spk[n] = [s[1] for s in spikes if s[0] == n]
#
#        return sol.t, sol.y.T, spk
        t = np.arange(0,T,dt)
        v = np.zeros((len(t),x0.shape[0]))
        v[0,:] = x0.copy()
        for i in range(1,len(t)):
            dvdt = ode_step(t[i],v[i-1,:],inp=inp,C=C,theta=theta,tau1=tau1,
                                g_l=g_l,tau2=tau2,v_rest=v_rest,
                                I_syn_avg=I_syn_avg)
            v[i,:] = v[i-1,:]+dt*dvdt
        
        
        spk = [[]]*N
        for n in range(N):
            spk[n] = [s[1] for s in spikes if s[0] == n]

        return t, v, spk
    
    @staticmethod
    def kadmon_network(parameters):
        if parameters['func'] == 'heaviside':
            phi = lambda a : np.heaviside(a,1)
        elif parameters['func'] == 'rect_tanh':
            phi = lambda a : np.concatenate((np.tanh(a[:,np.newaxis]),a[:,np.newaxis]*0),1).max(1)
        elif parameters['func'] == 'rect_lin':
            phi = lambda a : np.concatenate((a[:,np.newaxis],a[:,np.newaxis]*0),1).max(1)
            
                    
        def ode_step(t,x,J,inp):
            ϕ = phi(x)
            η = J @ ϕ
            dxdt = -x + η + inp(t)
            return dxdt
        
        J_mean = parameters['J_mean']
        J_std = parameters['J_std']
        EI_frac = parameters['J_std']
        N = parameters['N']
        T = parameters['T']
        EI_frac = parameters['EI_frac']
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
            t_lim = [parameters['t'].min(),parameters['t'].max()]
            t_eval = parameters['t']
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
            t_lim = [0,T]
            t_eval = None
        
         ## Initialize
        J = Simulator.randJ_EI_FC(N,J_mean,J_std,EI_frac)
        x0 = np.random.rand(N, 1)
        sol = solve_ivp(partial(ode_step,J=J,inp=inp),t_lim,x0.squeeze(),t_eval=t_eval)
        
        
        return sol.t, sol.y.T
    
    @staticmethod
    def luca_network(parameters,dt=.001):
        T = parameters['T']
        N = parameters['N']
        
        spikes = []
        
        Simulator.refr = np.zeros((N))
        Simulator.last_t = -T
        Simulator.current = np.zeros((N))
        
        def ode_step(t,v,tau_m,inp,tau_syn,v_rest,theta,J,f_mul,f_add,tau_arp,baseline):
            Simulator.refr = np.maximum(-0.001,Simulator.refr-(t-Simulator.last_t))
            v[Simulator.refr > 0] = v_rest[Simulator.refr > 0]
            
            fired = (v >= theta)
            v[v >= theta] = v_rest[v >= theta]
            
            [spikes.append((s,t)) for s in np.where(fired)[0]]
            Simulator.current *= np.exp((t-Simulator.last_t)*f_mul)
            Simulator.current[fired] += f_add[fired]
            Simulator.refr[fired] = tau_arp
            
            dvdt = -v/tau_m + J@Simulator.current + baseline + inp(t)
            Simulator.last_t = t
            return dvdt
            
        tau_m = parameters['tau_m'] 
        v_rest = parameters['v_rest'] 
        theta = parameters['theta']
        tau_syn = parameters['tau_syn'] 
        f_mul = parameters['f_mul'] 
        f_add = parameters['f_add']
        EI_frac = parameters['EI_frac']
        tau_arp = parameters['tau_arp']
        baseline = parameters['baseline']
        
        C = parameters['C']
        C_std = parameters['C_std']
        
        clusters_mean = parameters['clusters_mean']
        clusters_stds = parameters['clusters_stds']
        clusters_prob = parameters['clusters_prob']
        external_mean = parameters['external_mean']
        external_stds = parameters['external_stds']
        external_prob = parameters['external_prob']
        
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='nearest',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
        
        if 'J' in parameters.keys():
            J = parameters['J']
            C_size = None
        else:
            J, C_size = Simulator.clustered_connectivity(N,EI_frac,C=C,C_std=C_std,
                                                 clusters_mean=clusters_mean,clusters_stds=clusters_stds,clusters_prob=clusters_prob,
                                                 external_mean=external_mean,external_stds=external_stds,external_prob=external_prob,
                                                 visualize=False)
        
         ## Initialize
        x0 = (theta-v_rest)*np.random.rand(N)
        
        t = np.arange(-T,T,dt)
        v = np.zeros((len(t),x0.shape[0]))
        v[0,:] = x0.copy()
        for i in range(1,len(t)):
            dvdt = ode_step(t[i],v[i-1,:],tau_m=tau_m,inp=inp,
                            tau_syn=tau_syn,v_rest=v_rest,
                            theta=theta,J=J,f_mul=f_mul,
                            f_add=f_add,tau_arp=tau_arp,
                            baseline=baseline)
            v[i,:] = v[i-1,:]+dt*dvdt
        
        
        a = [(k, [x for _, x in g]) for k, g in groupby(sorted(spikes,key=itemgetter(0)), key=itemgetter(0))]
        spk = [[]]*N
        for n in range(len(a)):
            spk[a[n][0]] = a[n][1]

        return t, v, spk, spikes, x0, J, C_size
            
    
    @staticmethod
    def continuous_to_spktimes(x,times,threshold):
        integral = 0
        spktimes = []
        for t in range(len(x)):
            integral = np.nansum([integral,x[t]])
            if integral > threshold:
                integral = 0
                spktimes.append(times[t])
        return np.array(spktimes)
    
    @staticmethod
    def spktimes_to_rates(spk,n_bins=100,rng=(-1,1),sigma=.1):
        bins = np.linspace(rng[0],rng[1],n_bins)
        rate = np.zeros((n_bins,len(spk)))
        for s in range(len(spk)):
            rate[:,s] = np.exp(-(cdist(spk[s][:,np.newaxis],bins[:,np.newaxis])/sigma)**2).sum(0)
    
        return rate,bins
    
    @staticmethod
    def randJ_EI_FC(N,J_mean=np.array([[1,2],[1,1.8]])
                    ,J_std=np.ones((2,2)),EI_frac=0):
        
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
    
    @staticmethod
    def bipartite_connectivity(M,N,p,visualize=False):
        G = bipartite.random_graph(M, N, p)
        if visualize:
            nx.draw(G, with_labels=True)
            plt.show() 
        return convert_matrix.to_numpy_array(G)

    @staticmethod
    def erdos_renyi_connectivity(N,p,visualize=True):
        G = nx.erdos_renyi_graph(N,p)
        if visualize:
            nx.draw(G, with_labels=True)
            plt.show() 
        return convert_matrix.to_numpy_array(G)
    
    @staticmethod
    def normal_connectivity(N,g):
        return g*np.random.normal(loc=0.0, scale=1/N, size=(N,N))
    
    
    
    @staticmethod
    def show_connectivity(adjacency):
        G = nx.from_numpy_array(adjacency)
        nx.draw(G, with_labels=True)
        
        
    @staticmethod
    def dag_connectivity(N,p=.5,visualize=True,save=False):

        G=nx.gnp_random_graph(N,p,directed=True)
        DAG = nx.DiGraph([(u,v,{'weight':random.randint(0,10)}) for (u,v) in G.edges() if u<v])
        
        if visualize:
            nx.draw(DAG, with_labels=True)
            if save:
                plt.savefig('results\\J.png')
            else:
                plt.show()
            
        
        return convert_matrix.to_numpy_array(DAG)
    
    def clustered_connectivity(N,EI_frac=0,C=10,C_std=[.2,0],
                               clusters_mean=[[0.1838,-0.2582],[0.0754,-0.4243]],
                               clusters_stds=[[.0,.0],[.0,.0]],
                               clusters_prob=[[.2,.5],[.5,.5]],
                               external_mean=[[.0036,-.0258],[.0094,-.0638]],
                               external_stds=[[.0,.0],[.0,.0]],
                               external_prob=[[.2,.5],[.5,.5]],
                               visualize=False):
        def EI_block_diag(cs,vs):
            return np.hstack((
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[0,i]))*vs[0,0,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[0,i]))*vs[1,0,i] for i in range(len(cs[0]))]))) ,
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[1,i]))*vs[0,1,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[1,i]))*vs[1,1,i] for i in range(len(cs[0]))])))
            ))
        
        E = round(N*EI_frac)
        I = N - E
        
        
        c_size = np.round((np.array([[E,I]]).T/C)*np.array(C_std)[:,np.newaxis]*np.random.randn(2,C) + (np.array([[E,I]]).T/C)).astype(int)
        c_size[:,-1] = np.array([E,I]).T-c_size[:,:-1].sum(1)
        
        c_mean = np.zeros((2,2,C))
        c_prob = np.zeros((2,2,C))
        c_stds = np.zeros((2,2,C))
        
        
        c_mean[0,:,:] = np.vstack((clusters_mean[0][0]*c_size[0,:].mean()/c_size[0,:],clusters_mean[0][1]*c_size[0,:].mean()/c_size[0,:]))
        c_mean[1,:,:] = np.vstack((clusters_mean[1][0]*c_size[1,:].mean()/c_size[1,:],clusters_mean[1][1]*c_size[1,:].mean()/c_size[1,:]))
        
        c_prob[0,:,:] = np.vstack((clusters_prob[0][0]+np.zeros((C)),clusters_prob[0][1]+np.zeros((C))))
        c_prob[1,:,:] = np.vstack((clusters_prob[1][0]+np.zeros((C)),clusters_prob[1][1]+np.zeros((C))))
        
        c_stds[0,:,:] = np.vstack((clusters_stds[0][0]+np.zeros((C)),clusters_stds[0][1]+np.zeros((C))))
        c_stds[1,:,:] = np.vstack((clusters_stds[1][0]+np.zeros((C)),clusters_stds[1][1]+np.zeros((C))))
        
        e_size = c_size.sum(1)[:,np.newaxis]
        e_prob = np.array(external_prob)[:,:,np.newaxis]
        e_mean = np.array(external_mean)[:,:,np.newaxis]
        e_stds = np.array(external_stds)[:,:,np.newaxis]
        
        JC_prob = EI_block_diag(c_size,c_prob)
        JC_mean = EI_block_diag(c_size,c_mean)
        JC_stds = EI_block_diag(c_size,c_stds)
        
        JC_mask = EI_block_diag(c_size,np.ones((2,2,C)))
        
        JE_prob = EI_block_diag(e_size,e_prob)*(1-JC_mask)
        JE_mean = EI_block_diag(e_size,e_mean)*(1-JC_mask)
        JE_stds = EI_block_diag(e_size,e_stds)*(1-JC_mask)
        
        J_prob = JC_prob + JE_prob
        J_mean = JC_mean + JE_mean
        J_stds = JC_stds + JE_stds
        
        J = np.random.binomial(n=1,p=J_prob)*(np.random.randn(N,N)*J_stds+J_mean)
        
        if visualize:
            plt.imshow(J)
            
        J[:E,:E] = np.maximum(0,J[:E,:E])
        J[:E,E:] = np.minimum(0,J[:E,E:])
        J[E:,:E] = np.maximum(0,J[E:,:E])
        J[E:,E:] = np.minimum(0,J[E:,E:])
        
        
        
        return J, c_size
