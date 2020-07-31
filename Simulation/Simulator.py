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
    def larry_model(parameters):
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
    def hansel_network(parameters):
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
        
         ## Initialize
        x0 = np.random.rand(N,1)*20-40
        sol = solve_ivp(partial(ode_step,inp=inp,C=C,theta=theta,tau1=tau1,
                                g_l=g_l,tau2=tau2,v_rest=v_rest,
                                I_syn_avg=I_syn_avg),[0,T],x0.squeeze())
        
        spk = [[]]*N
        for n in range(N):
            spk[n] = [s[1] for s in spikes if s[0] == n]

        return sol.t, sol.y.T, spk
    
    
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
    def randJ_EI_FC(N,J_mean=np.array([[1,2],[1,1.8]])
                    ,J_std=np.ones((2,2)),EI_frac=0):
        
        E = int(N*EI_frac)
        I = int(N*(1-EI_frac))
        
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