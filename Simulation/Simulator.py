# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin
"""

import random

import numpy as np
import networkx as nx
from scipy.io import savemat
from functools import partial
from scipy import interpolate
import matplotlib.pyplot as plt
from networkx import convert_matrix
from sklearn.decomposition import PCA
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d as intp
from networkx.algorithms import bipartite
from scipy.linalg import block_diag
from itertools import groupby
from operator import itemgetter

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
    def rossler_downstream(parameters,save_data=False,file=None):
        def ode_step(t,x,alpha,beta,gamma,inp,lambd,J):
            dxdt_u = [-(x[1]+x[2]),x[0]+alpha*x[1],beta+x[2]*(x[0]-gamma)]
            dxdt_d = -lambd*x[3:] + 10*np.tanh(J@x)
            dxdt = np.concatenate((dxdt_u,dxdt_d)) + inp(t)
            return dxdt
        
        p = parameters['bernoulli_p']
        N = parameters['N']
        g_r = parameters['g_r']
        g_i = parameters['g_i']
        lambd = parameters['lambda']
        
        if 'J' in parameters.keys():
            J = parameters['J']
        else:
            J1 = g_i*Simulator.bipartite_connectivity(3,N-3,p,visualize=False)[3:,:3]
            J2 = g_r*Simulator.normal_connectivity(N-3,1)
            J = np.hstack((J1,J2))
        
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='nearest',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
            
        if 't_eval' in parameters.keys():
            t_eval = parameters['t_eval']
        else:
            t_eval = None
            
        ## Initialize
        if 'x0' in parameters.keys():
            x0 = parameters['x0']
        else:
            x0 = 5*np.random.rand(N,1)
            
        sol = solve_ivp(partial(ode_step,alpha=parameters['alpha'],beta=parameters['beta'],gamma=parameters['gamma'],
                                J=J,inp=inp,lambd=lambd),
                        [0,parameters['T']],x0.squeeze(),t_eval=t_eval)
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y,'J':J})
    
        return sol.t, sol.y.T, J
    
    @staticmethod
    def downstream_network(I,t,parameters):
            
        def ode_step(t, x, J, inp, g_i, g_r, lambd, inp_ext):
            dxdt=-lambd*x + 10*np.tanh(g_r*J@x+g_i*inp(t)+inp_ext(t))
            return dxdt
        
        p = parameters['bernoulli_p']
        N = parameters['N']
        g_r = parameters['g_r']
        g_i = parameters['g_i']
        lambd = parameters['lambda']
        
        noise = np.random.randn(I.shape[0],I.shape[1])*parameters['noise_std']
        
        if 'U_J' in parameters.keys():
            U_J = parameters['U_J']
        else:
            U_J = Simulator.bipartite_connectivity(I.shape[1],N,p,visualize=False)[I.shape[1]:,:I.shape[1]]
        
        if 'J' in parameters.keys():
            J = parameters['J']
        else:
            J = Simulator.normal_connectivity(N,1)
        
        I = I + noise
        
        inp = interpolate.interp1d(t,U_J@I.T,kind='linear',bounds_error=False)
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp_ext = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
        else:
            inp_ext = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')

        
        ## Initialize
        x0 = np.random.rand(N, 1)
        sol = solve_ivp(partial(ode_step,J=J,inp=inp,g_i=g_i,g_r=g_r,lambd=lambd,inp_ext=inp_ext),
                        [t.min(),t.max()],x0.squeeze(),t_eval=t)
    
        x  = sol.y
        return t, x.T, J, U_J
        
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
    def luca_network(parameters,dt=.001,save_data=False,file=None):
        T = parameters['T']
        N = parameters['N']
        
        spikes = []
        
        Simulator.refr = np.zeros((N))
        Simulator.last_t = -T
        Simulator.current = np.zeros((N))
        
        def ode_step(t,v,tau_m,inp,tau_syn,v_rest,theta,J,f_mul,f_add,tau_arp,baseline,dt):
            Simulator.refr = np.maximum(-0.001,Simulator.refr-(t-Simulator.last_t))
            v[Simulator.refr > 0] = v_rest[Simulator.refr > 0]
            
            fired = (v >= theta)
            v[fired] = v_rest[fired]
            
            [spikes.append((s,t)) for s in np.where(fired)[0]]
            Simulator.current *= np.exp((t-Simulator.last_t)*f_mul)
            Simulator.current[fired] += f_add[fired]
            Simulator.refr[fired] = tau_arp
            
            dvdt = -v/tau_m + J@Simulator.current + baseline + inp(t)
            Simulator.last_t = t
            
            v[~fired] = v[~fired]+dt*dvdt[~fired]
            
            return v
            
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
            C_size = np.nan
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
            v[i,:] = ode_step(t[i],v[i-1,:],tau_m=tau_m,inp=inp,
                            tau_syn=tau_syn,v_rest=v_rest,
                            theta=theta,J=J,f_mul=f_mul,
                            f_add=f_add,tau_arp=tau_arp,
                            baseline=baseline,dt=dt)
            
        
        
        a = [(k, [x for _, x in g]) for k, g in groupby(sorted(spikes,key=itemgetter(0)), key=itemgetter(0))]
        spk = [[]]*N
        for n in range(len(a)):
            spk[a[n][0]] = a[n][1]
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':t,'v':v,'J':J,'C_size':C_size,
                                 'spk':spk,'spikes':spikes})

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
    def spktimes_to_rates(spk,n_bins=100,rng=(-1,1),sigma=.1,method='gaussian',save_data=False,file=None):
        bins = np.linspace(rng[0],rng[1],n_bins)
        rate = np.zeros((n_bins,len(spk)))
        bin_edges = np.linspace(rng[0],rng[1],n_bins+1)
        
        for s in range(len(spk)):
            if method == 'gaussian':
                rate[:,s] = np.exp(-(cdist(spk[s][:,np.newaxis],bins[:,np.newaxis])/sigma)**2).sum(0)
            elif method == 'counts':
                rate[:,s] = np.histogram(spk[s],bin_edges)[0]
                
        if save_data:
            savemat(file+'.mat',{'spk':spk,'n_bins':n_bins,'rng':rng,'sigma':sigma,
                                 'method':method,'bins':bins,'rate':rate,'bin_edges':bin_edges})
                
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
    def show_clustered_connectivity(adjacency,clusters,exc,save=False,file=None):
        G = nx.from_numpy_array(adjacency,create_using=nx.DiGraph)
        weights = nx.get_edge_attributes(G,'weight').values()
        
        G_ = nx.from_numpy_array(np.ones((len(clusters),len(clusters))))
        pos = np.array(list(nx.spring_layout(G_, iterations=100).values()))
        pos = np.repeat(pos, clusters, axis=0)        
        
        rpos = np.hstack([np.array([.08*np.cos(np.linspace(0,2*np.pi,1+clusters[i])[:-1]), 
                                    .08*np.sin(np.linspace(0,2*np.pi,1+clusters[i])[:-1])]) 
                for i in range(len(clusters))])
                
        plt.figure(figsize=(10,10))
        
        node_color = np.array([[0,0,1,.5]]*exc + [[1,0,0,.5]]*(G.number_of_nodes()-exc))
        
        options = {
            'node_color': node_color,
            'edgecolors': 'k',
            'node_size': 300,
            'width': 2*np.array(list(weights)),
            'arrowstyle': '-|>',
            'arrowsize': 15,
            'font_size':10, 
            'font_family':'fantasy',
            'connectionstyle':"arc3,rad=-0.1",
        }
        
        nx.draw(G, pos=list(pos+rpos.T), with_labels=True, arrows=True, **options)
        
        if save:
            plt.savefig(file+'.eps',format='eps')
            plt.savefig(file+'.png',format='png')
            plt.close('all')
        else:
            plt.show()
        
    @staticmethod
    def show_downstream_connectivity(adjacency,fontsize=20,save=False,file=None):
        G = nx.from_numpy_array(adjacency,create_using=nx.DiGraph)
        weights = nx.get_edge_attributes(G,'weight').values()
        
        node_color = np.array([[0,0,1,.5]]*3 + [[1,0,1,.5]]*(G.number_of_nodes()-3))
        
        if adjacency.shape[0] == 10:
            options = {
                'node_color': node_color,
                'edgecolors': 'k',
                'node_size': 3000,
                'width': 20*np.array(list(weights)),
                'arrowstyle': '-|>',
                'arrowsize': 20,
                'font_size':fontsize, 
                'font_family':'fantasy',
                'connectionstyle':'arc3,rad=0',
            }
            plt.figure(figsize=(5,8))
        elif adjacency.shape[0] > 100:
            node_size = np.concatenate((np.ones((3,1)),np.zeros((G.number_of_nodes()-3,1))))
            options = {
                'node_color': node_color,
                'edgecolors': 'k',
                'node_size': node_size*2500+500,
                'width': 1*np.array(list(weights)),
                'arrowstyle': '-|>',
                'arrowsize': 20,
                'font_size':fontsize, 
                'font_family':'fantasy',
                'connectionstyle':'arc3,rad=0',
            }
            plt.figure(figsize=(15,8))

        pos = nx.bipartite_layout(G,set(np.arange(3)),align='horizontal')
        
        pos = np.array(list(pos.values()))
        
        m1 = pos[:3,:].mean(0)
        m2 = pos[3:,:].mean(0)
        
        pos[:3,:] = m1[None,:] + .1*np.array([np.sin(np.linspace(0,2*np.pi,4)[:-1]), 
                                           np.cos(np.linspace(0,2*np.pi,4)[:-1])]).T
                
        pos[3:,:] = m2[None,:] + .2*np.array([np.sin(np.linspace(0,2*np.pi,G.number_of_nodes()-2)[:-1]), 
                                           np.cos(np.linspace(0,2*np.pi,G.number_of_nodes()-2)[:-1])]).T
        
        
        nx.draw(G, pos=pos, with_labels=True, arrows=True, **options)
        
        if save:
            plt.savefig(file+'.eps',format='eps')
            plt.savefig(file+'.png',format='png')
            plt.close('all')
        else:
            plt.show()
        
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
    
    
    @staticmethod
    def stimulation_protocol(C_size,time_st,time_en,N,n_record,stim_d,rest_d,
                             feasible,amplitude,repetition=1,fraction_stim=.8,
                             fontsize=20,visualize=True,save=False,file=None,
                             save_data=False):
        
        c_range = np.cumsum(np.hstack((0,C_size)))
        t_stim = np.linspace(time_st,time_en,repetition*int((stim_d+rest_d)/stim_d)*(len(c_range)-1))
        I = np.zeros((len(t_stim),N))
        
        stimulated = []
        for c in range(len(c_range)-1):
            sz = int(((c_range[c+1]-c_range[c])*fraction_stim).round())
            rand_sample = np.random.choice(np.arange(c_range[c],c_range[c+1]),
                   size=sz,replace=False).astype(int)
            stimulated.append(rand_sample)
            
        recorded = []
        for c in range(len(c_range)-1):
            rand_sample = stimulated[c]
            rand_sample = rand_sample[np.where(feasible[rand_sample])[0]]
            recorded += list(rand_sample[:n_record])
        
        
        for r in range(repetition):
            clusters = np.arange(len(c_range)-1)
            np.random.shuffle(clusters)
            
            for c_idx,c in enumerate(clusters):
                time_idx = r*len(clusters)+c_idx
                
                rand_sample = stimulated[c]
                d1 = int(int((stim_d+rest_d)/stim_d)*stim_d)
                d2 = int((stim_d+rest_d)/stim_d)
                I[d2*time_idx+d2//2:d2*time_idx+d2//2+d1,rand_sample] = amplitude[rand_sample]
        
        if visualize:
            plt.figure()
            plt.imshow(I.T,aspect='auto',interpolation='none', extent=[time_st,time_en,0,N],origin='lower')
            plt.xlabel('time',fontsize=fontsize)
            plt.ylabel('Neurons',fontsize=fontsize)
            plt.title('Stimulation Protocol',fontsize=fontsize)
            
            if save:
                plt.savefig(file+'stim-protocol.png')
                plt.close('all')
            else:
                plt.show()
                
                
        if save_data:
            savemat(file+'.mat',{'C_size':C_size,'time_st':time_st,'time_en':time_en,
                                 'N':N,'n_record':n_record,'stim_d':stim_d,
                                 'rest_d':rest_d,'feasible':feasible,'amplitude':amplitude,
                                 'repetition':repetition,'fraction_stim':fraction_stim,
                                 'I':I,'t_stim':t_stim,'recorded':recorded,'stimulated':stimulated})
                
        return I,t_stim,recorded,stimulated
    
    
    @staticmethod
    def divide_clusters(clusters,C=10,C_std=.1):
        clusters_ = []
        for c in clusters:
            c_ = np.round((c/C)*C_std*np.random.randn(C)) + (c/C).astype(int)
            c_[-1] = c - c_[:-1].sum()
            clusters_ += list(c_.astype(int))
        
        return np.array(clusters_)