# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: Amin
"""

from networkx.algorithms import bipartite
from scipy.spatial.distance import cdist
from scipy.integrate import solve_ivp
from scipy.linalg import block_diag
from networkx import convert_matrix
import matplotlib.pyplot as plt
from operator import itemgetter
from functools import partial
from scipy import interpolate
from itertools import groupby
from scipy.io import savemat
from functools import reduce
import networkx as nx
import numpy as np
import random
import scipy

class Simulator(object):
    @staticmethod
    def lorenz_network(parameters={'T':1000,'s':10,'r':28,'b':2.667},save_data=False,file=None):
        def ode_step(t,x,s,r,b,inp):
            dxdt = [s*(x[1] - x[0]),
                    r*x[0] - x[1] - x[0]*x[2],
                    x[0]*x[1] - b*x[2]] + inp(t)
            return dxdt
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
            
        if 't_eval' in parameters.keys():
            t_eval = parameters['t_eval']
        else:
            t_eval = None
            
        x0 = np.random.rand(3, 1)
        sol = solve_ivp(partial(ode_step,s=parameters['s'],r=parameters['r'],b=parameters['b'],inp=inp),
                        [0,parameters['T']],x0.squeeze(),t_eval=t_eval)
        
        if 'rotation' in parameters.keys():
            sol.y = parameters['rotation']@sol.y
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y})
            
        return sol.t, sol.y.T


        
    @staticmethod
    def rossler_network(parameters={'T':1000,'alpha':.2,'beta':.2,'gamma':5.7},save_data=False,file=None):

        """Simulate data from rossler attractor 
        https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
            
        dx/dt = [-x[1]-x[2],
                    x[0]+alpha*x[1],
                    beta+x[2]*(x[0]-gamma)]
        
        Args:
            parameters (dict): Parameters of the rossler attractor 
                (time,alpha,beta,gamma)
            save_data (bool): If True the simulated data will be saved
            file (string): File address for saving the data
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (3xT) rossler data
        
        """
    
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='linear',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
            
        def ode_step(t,x,alpha,beta,gamma,inp):
            dxdt = [-(x[1]+x[2]),
                x[0]+alpha*x[1],
                beta+x[2]*(x[0]-gamma)] + inp(t)
            return dxdt
        
        if 't_eval' in parameters.keys():
            t_eval = parameters['t_eval']
        else:
            t_eval = None
        
        x0 = np.random.rand(3, 1)
        sol = solve_ivp(partial(ode_step,alpha=parameters['alpha'],beta=parameters['beta'],gamma=parameters['gamma'],inp=inp),
                        [0,parameters['T']],x0.squeeze(),t_eval=t_eval)
        
        if 'rotation' in parameters.keys():
            sol.y = parameters['rotation']@sol.y
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y})
            
        return sol.t, sol.y.T
    
    @staticmethod
    def rossler_downstream(parameters,save_data=False,file=None):
        """Simulate data from rossler attractor that is unidirectionally 
            connected to a downstream rate network
            https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
            
            dx/dt = [-x[1]-x[2],
                    x[0]+alpha*x[1],
                    beta+x[2]*(x[0]-gamma)]
            dy/dt -lambd*x + 10*tanh(J@x)
        
        Args:
            parameters (dict): Parameters of the rossler attractor 
                (time,alpha,beta,gamma,lambda) for dynamics and
                (p) for the connectivity matrix
            save_data (bool): If True the simulated data will be saved
            file (string): File address for saving the data
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (N+3xT) rossler data
            numpy.ndarray: Randomly sampled (N+3,N+3) connectivity matrix
        
        """
        
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
    def thomas_attractor(parameters={'T':1000,'b':.198},save_data=False,file=None):

        def ode_step(t,x,b):
            dxdt = [np.sin(x[1])-b*x[0],np.sin(x[2])-b*x[1],np.sin(x[0])-b*x[2]]
            return dxdt
        x0 = np.random.randn(3)#np.array([2.7880,2.5856,2.3069])
        sol = solve_ivp(partial(ode_step,b=parameters['b']),[0,parameters['T']],x0.squeeze())
        
        if 'rotation' in parameters.keys():
            sol.y = parameters['rotation']@sol.y
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y})

        return sol.t, sol.y.T

    @staticmethod
    def rossler2(parameters={'T':1000,'alpha':.1,'beta':.1,'gamma':14},save_data=False,file=None,plot=True):
        """
        test
        """
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
        
        if 'rotation' in parameters.keys():
            sol.y = parameters['rotation']@sol.y
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y})


        return sol.t, sol.y.T


    @staticmethod
    def langford_attractor(parameters={'T':1000,'a':.95,'b':.7,'c':.6,'d':3.5,'e':.25,'f':.1},save_data=False,file=None):
        """"
        #  Langford, William F. "Numerical studies of torus bifurcations." In Numerical Methods for Bifurcation Problems, pp. 285-295. Birkhäuser, Basel, 1984. 

        x0 = 0.1 y0 = 0 z0 = 0 
        ε = 0.3 α = 0.95 ɣ = 0.6 δ = 3.5 β = 0.7 ζ = 0.1  #### or rather ε = 0.25?
        t = 15000 Step = 25
        dx = (z-β)*x-δ*y
        dy = δ*x+(z-β)*y
        dz = ɣ+α*z-(z**3/3)-(x**2+y**2)*(1+ε*z)+ζ*z*x**3
        
        """ 
        def ode_step(t,x,a=.95,b=.7,c=.6,d=3.5,e=.25,f=.1):
            dxdt = [(x[2]-b)*x[0]-d*x[1],
                    d*x[0]+(x[2]-b)*x[1],
                    c+a*x[2]-x[2]**3/3-(x[0]**2+x[1]**2)*(1+e*x[2])+f*x[2]*(x[0]**3)]
                    #c+a*x[2]-x[2]**3/3-x[2]+f*x[2]*(x[0]**3)]
            return dxdt

        x0 = np.array([0.1,0,0])#-.6502,0.9267,1.4722])
        sol = solve_ivp(partial(ode_step,a=parameters['a'],b=parameters['b'],c=parameters['c'],d=parameters['d'],e=parameters['e'],f=parameters['f']),[0,parameters['T']],x0.squeeze())
        
        if 'rotation' in parameters.keys():
            sol.y = parameters['rotation']@sol.y
            
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y})
            
        return sol.t, sol.y.T

    @staticmethod
    def downstream_network(I,t,parameters,save_data=False,file=None):
        """Simulate data from a downstream network where the network is driven
            by external input
            
            dy/dt -lambd*x + 10*tanh(J@x) where x is the input
        
        Args:
            I (numpy.ndarray): External input
            t (numpy.ndarray): Time points for the sampled external input
            
            save_data (bool): If True the simulated data will be saved
            file (string): File address for saving the data
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (3xT) rossler data
            numpy.ndarray: Randomly sampled (N,N) recurrent connectivity matrix
            numpy.ndarray: Randomly sampled (N,3) input connectivity matrix
            
        """
        
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
        
        if save_data:
            savemat(file+'.mat',{'parameters':parameters,'t':sol.t,'y':sol.y,'J':J,'noise':noise,'U_J':U_J})
            
        x  = sol.y
        return t, x.T, J, U_J
        
    @staticmethod
    def larry_network(parameters):
        """Simulate data from a network studied by L. Abbott et al
            https://arxiv.org/abs/0912.3832
            
            dx/dt = 1/tau*(-x+J@r+input)
        
        Args:
            parameters (dict): Parameters of the network 
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (NxT) data
            numpy.ndarray: External input time series
        
        """
        
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
        """Simulate data from a network studied by D. Hansel et al
            https://www.mitpressjournals.org/doi/pdf/10.1162/089976698300017845
            
            dx/dt = 1/tau*(-x+J@r+input)
        
        Args:
            parameters (dict): Parameters of the network 
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (NxT) data
            array: array of pairs (channel,time) spikes generated by the 
                network
        
        """
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
        """Simulate data from a network studied by J. Kadmon et al
            https://journals.aps.org/prx/pdf/10.1103/PhysRevX.5.041030
            
        Args:
            parameters (dict): Parameters of the network 
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (NxT) data
        
        """
        
        if parameters['func'] == 'heaviside':
            phi = lambda a : np.heaviside(a,1)
        elif parameters['func'] == 'rect_tanh':
            phi = lambda a : np.concatenate((np.tanh(a[:,np.newaxis]),a[:,np.newaxis]*0),1).max(1)
        elif parameters['func'] == 'rect_lin':
            phi = lambda a : np.concatenate((a[:,np.newaxis],a[:,np.newaxis]*0),1).max(1)
            
                    
        def ode_step(t,x,J,inp):
            phi = phi(x)
            η = J @ phi
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
        """Simulate data from a network studied by L. Mazzucato et al
            https://www.biorxiv.org/content/10.1101/2020.04.07.030700v1
            
        Args:
            parameters (dict): Parameters of the network 
            
        Returns:
            numpy.array: Time for the generated time series
            numpy.ndarray: Simulated (NxT) voltages
            array: Array of pairs (channel,time) spikes generated by the 
                network
            array: Spikes in a flat data structure
            numpy.ndarray: Initialized network voltages
            numpy.ndarray: Connectivity matrix (NxN)
            array: Size of the clusters
        
        """
        
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
        
        
        if 't' in parameters.keys() and 'I' in parameters.keys() and 'I_J' in parameters.keys():
            inp = interpolate.interp1d(parameters['t'],(parameters['I']@parameters['I_J']).T,kind='nearest',bounds_error=False)
        else:
            inp = interpolate.interp1d([0,1],[0,0],kind='nearest',fill_value='extrapolate')
        
        if 'J' in parameters.keys():
            J = parameters['J']
            if 'C_size' in parameters.keys():
                C_size = parameters['C_size']
            else:
                C_size= np.nan
        else:
            C = parameters['C']
            C_std = parameters['C_std']
            
            clusters_mean = parameters['clusters_mean']
            clusters_stds = parameters['clusters_stds']
            clusters_prob = parameters['clusters_prob']
            external_mean = parameters['external_mean']
            external_stds = parameters['external_stds']
            external_prob = parameters['external_prob']
            
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
        """Conversion of continous signals to spike times for testing the ISI
            delay embedding framework
            
        Args:
            x (numpy.ndarray): Continous signal (1xT)
            times (numpy.ndarray): Sampling times from the signal
            threshold (float): Threshold used for generating spikes
            
        Returns:
            array: Spike times
        
        """
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
        """Conversion of spike times to rates
            
        Args:
            spk (array): Spike times to be converted to rates
            n_bins (numpy.ndarray): Sampling times from the signal
            rng (float): Threshold used for generating spikes
            sigma (float)
            method (string): Smoothing method, choose from ('gaussian','counts')
            save_data (bool): If True the generated rates will be saved in a mat
                file
            file (string): File address used for saving the mat file
        Returns:
            numpy.ndarray: Rates that are converted from input spikes
            numpy.ndarray: Bins used for windowing the spikes
        """
        
        bins = np.linspace(rng[0],rng[1],n_bins)
        rate = np.zeros((n_bins,len(spk)))
        bin_edges = np.linspace(rng[0],rng[1],n_bins+1)
        bin_len = (rng[1]-rng[0])/n_bins
        filt = np.exp(-(np.arange(-3*sigma/bin_len,3*sigma/bin_len,(rng[1]-rng[0])/n_bins)/sigma)**2)
        
        for s in range(len(spk)):
            print(s)
            rate[:,s] = np.histogram(spk[s],bin_edges)[0]
            if method == 'gaussian':
                rate[:,s] = scipy.signal.convolve(rate[:,s],filt,mode='same')
                
        if save_data:
            savemat(file+'.mat',{'spk':spk,'n_bins':n_bins,'rng':rng,'sigma':sigma,
                                 'method':method,'bins':bins,'rate':rate,'bin_edges':bin_edges})
                
        return rate,bins
    
    @staticmethod
    def randJ_EI_FC(N,J_mean=np.array([[1,2],[1,1.8]])
                    ,J_std=np.ones((2,2)),EI_frac=0):
        """Create random excitatory inhibitory connectivity matrix from input 
            statistics
            
        Args:
            N (integer): Total number of nodes in the network
            J_mean (numpy.ndarray): 2x2 array of the mean for excitatory and 
                inhibitory population connectivity
            J_std (numpy.ndarray): 2x2 array of the standard deviation for 
                excitatory and inhibitory population connectivity
            EI_frac (float): Fraction of excitatory to inhibitory neurons 
                (between 0,1)
        
        Returns:
            array: Randomly generated matrix
        
        """
        
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
        """Create random bipartite connectivity matrix
            https://en.wikipedia.org/wiki/Bipartite_graph
            
        Args:
            M (integer): Number of nodes in the first partite
            N (integer): Number of nodes in the second partite
            p (float): Connection probability (between 0,1)
            visualize (bool): If true the graph will be visualized
        
        Returns:
            array: Randomly generated matrix
        
        """
        G = bipartite.random_graph(M, N, p)
        if visualize:
            nx.draw(G, with_labels=True)
            plt.show() 
        return convert_matrix.to_numpy_array(G)

    @staticmethod
    def erdos_renyi_connectivity(N,p,visualize=True):
        """Create random Erdos Renyi connectivity matrix
            https://en.wikipedia.org/wiki/Erd%C5%91s%E2%80%93R%C3%A9nyi_model
            
        Args:
            N (integer): Number of nodes in the network
            p (float): Connection probability (between 0,1)
            visualize (bool): If true the graph will be visualized
        
        Returns:
            numpy.ndarray: Randomly generated matrix
        
        """
        
        G = nx.erdos_renyi_graph(N,p)
        if visualize:
            nx.draw(G, with_labels=True)
            plt.show() 
        return convert_matrix.to_numpy_array(G)
    
    @staticmethod
    def normal_connectivity(N,g):
        """Normal random connectivity matrix
            
        Args:
            N (integer): Number of nodes in the network
            g (float): Connection strength
        
        Returns:
            numpy.ndarray: Randomly generated matrix
        
        """
        
        return g*np.random.normal(loc=0.0, scale=1/N, size=(N,N))
    
    
    @staticmethod
    def show_clustered_connectivity(adjacency,clusters,exc,save=False,file=None):
        """Visualize clustered connectivity graph
            
        Args:
            adjacency (matrix): Adjacency matrix of the connectivity
            clusters (float): Array of cluster sizes
            exc (integer): Number of excitatory nodes for coloring
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot
        
        """
        
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
        """Visualize downstream connectivity graph
            
        Args:
            adjacency (matrix): Adjacency matrix of the connectivity
            fontsize (float): Font size used for plotting
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot
        
        """
        
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
    def dag_connectivity(N,p=.5,visualize=True,save=False,file=None):
        """Directed acyclic graph random connectivity matrix
            https://en.wikipedia.org/wiki/Directed_acyclic_graph
            
        Args:
            N (integer): Number of nodes in the network
            p (float): Connection probability (between 0,1) look at the 
                documentation of gnp_random_graph
            visualize (bool): If true the graph will be visualized
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot

        Returns:
            numpy.ndarray: Randomly generated matrix
        
        """
        
        G=nx.gnp_random_graph(N,p,directed=True)
        DAG = nx.DiGraph([(u,v,{'weight':random.randint(0,10)}) for (u,v) in G.edges() if u<v])
        
        if visualize:
            nx.draw(DAG, with_labels=True)
            if save:
                plt.savefig(file+'.eps',format='eps')
                plt.savefig(file+'.png',format='png')
                plt.close('all')
            else:
                plt.show()
            
        
        return convert_matrix.to_numpy_array(DAG)
    
    @staticmethod
    def geometrical_connectivity(N,decay=1,EI_frac=0,mean=[[0.1838,-0.2582],[0.0754,-0.4243]],
                                   prob=[[.2,.5],[.5,.5]],visualize=False,save=False,file=None):
        """Create random  connectivity graph that respects the geometry of the 
            nodes in which nodes that are closer are more likely to be connected
            
        Args:
            N (integer): Number of nodes in the network
            decay (float): Decay of the weight strength as a function of physical
                distance
            EI_frac (float): Fraction of excitatory to inhibitory nodes
            mean (array): 2x2 array representing the mean of the EE/EI/IE/II 
                population
            prob (array): 2x2 array representing the probability of the EE/EI/IE/II 
                population
            visualize (bool): If true the graph will be visualized
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot


        Returns:
            numpy.ndarray: Randomly generated matrix
            array: Location of the nodes in the simulated physical space
        
        """
        
        def EI_block_diag(cs,vs):
            return np.hstack((
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[0,i]))*vs[0,0,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[0,i]))*vs[1,0,i] for i in range(len(cs[0]))]))) ,
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[1,i]))*vs[0,1,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[1,i]))*vs[1,1,i] for i in range(len(cs[0]))])))
            ))
        
        E = round(N*EI_frac)
        I = N - E
        
        X = np.random.rand(N,2)
        
        J_prob = EI_block_diag(np.array([[E],[I]]),np.array(prob)[:,:,np.newaxis])
        J = np.exp(-cdist(X,X)**2/decay)*np.random.binomial(n=1,p=J_prob)
        
        J[:E,:E] = mean[0][0]*J[:E,:E]/J[:E,:E].mean()
        J[:E,E:] = mean[0][1]*J[:E,E:]/J[:E,:E].mean()
        J[E:,:E] = mean[1][0]*J[E:,:E]/J[:E,:E].mean()
        J[E:,E:] = mean[1][1]*J[E:,E:]/J[:E,:E].mean()
        
        
        if visualize:
            plt.figure(figsize=(10,10))
        
            node_color = np.array([[0,0,1,.5]]*E + [[1,0,0,.5]]*I)
            G = nx.from_numpy_array(J,create_using=nx.DiGraph)
            weights = nx.get_edge_attributes(G,'weight').values()
            
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
        
            print(len(list(X)))
            nx.draw(G, pos=list(X), with_labels=True, arrows=True, **options)
        
            if save:
                plt.savefig(file+'.eps',format='eps')
                plt.savefig(file+'.png',format='png')
                plt.savefig(file+'.pdf',format='pdf')
                plt.close('all')
            else:
                plt.show()
            
        
        return J,X
    
    @staticmethod
    def clustered_connectivity(N,EI_frac=0,C=10,C_std=[.2,0],
                               clusters_mean=[[0.1838,-0.2582],[0.0754,-0.4243]],
                               clusters_stds=[[.0,.0],[.0,.0]],
                               clusters_prob=[[.2,.5],[.5,.5]],
                               external_mean=[[.0036,-.0258],[.0094,-.0638]],
                               external_stds=[[.0,.0],[.0,.0]],
                               external_prob=[[.2,.5],[.5,.5]],
                               external=None,
                               visualize=False,c_size=None):
        """Create random clustered inhibitory excitatory connectivity graph 
            
        Args:
            N (integer): Number of nodes in the network
            EI_frac (float): Fraction of excitatory to inhibitory nodes
            clusters_mean (array): 2x2 array representing the connection mean
                for in cluster connections (EE/EI/IE/EE)
            clusters_stds (array): 2x2 array representing the connection standard 
                deviation for in cluster connections
            clusters_prob (array): 2x2 array representing the connection probability
                for in cluster connections
            external_mean (array): 2x2 array representing the connection mean
                for out of cluster connections
            external_stds (array): 2x2 array representing the connection standard 
                deviation for out of cluster connections
            external_prob (array): 2x2 array representing the connection probability
                for out of cluster connections
            external (string): Out of cluster connectivity pattern, choose from
                ('cluster-block','cluster-column','random')
            visualize (bool): If true the graph will be visualized
            c_size (array): The number of nodes in each cluster (pre-given)


        Returns:
            numpy.ndarray: Randomly generated matrix
            array: Array of number of nodes in each cluster, first row 
                corresponds to excitatory and second row corresponds to
                inhibitory
        
        """
        
        def EI_block_diag(cs,vs):
            return np.hstack((
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[0,i]))*vs[0,0,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[0,i]))*vs[1,0,i] for i in range(len(cs[0]))]))) ,
            np.vstack((block_diag(*[np.ones((cs[0,i],cs[1,i]))*vs[0,1,i] for i in range(len(cs[0]))]),
                       block_diag(*[np.ones((cs[1,i],cs[1,i]))*vs[1,1,i] for i in range(len(cs[0]))])))
            ))
        
        
        
        if c_size is None:
            E = round(N*EI_frac)
            I = N - E
            c_size = np.round((np.array([[E,I]]).T/C)*np.array(C_std)[:,np.newaxis]*np.random.randn(2,C) + (np.array([[E,I]]).T/C)).astype(int)
            c_size[:,-1] = np.array([E,I]).T-c_size[:,:-1].sum(1)
        else:
            E,I = c_size.sum(1)
        
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
        
        if external=='cluster-block':
            jc_mask = EI_block_diag(np.ones((2,C),dtype=int),np.ones((2,2,C)))
            je_mean = EI_block_diag(np.array([[C],[C]],dtype=int),e_mean)*(1-jc_mask)
            je_stds = EI_block_diag(np.array([[C],[C]],dtype=int),e_stds)*(1-jc_mask)
            je = (np.random.randn(2*C,2*C)*je_stds+je_mean)*(1-jc_mask)
            JE_mean = je.repeat(c_size.flatten(),axis=0).repeat(c_size.flatten(),axis=1)
        elif external == 'cluster-column':
            jc_mask = EI_block_diag(np.ones((2,C),dtype=int),np.ones((2,2,C)))
            print((np.random.randn(1,C)*e_stds[0,0]+e_mean[0,0]).repeat(C,axis=0).shape)
            je = np.hstack((
                 np.vstack(((np.random.randn(1,C)*e_stds[0,0]+e_mean[0,0]).repeat(C,axis=0),
                           (np.random.randn(1,C)*e_stds[1,0]+e_mean[1,0]).repeat(C,axis=0))),
                 np.vstack(((np.random.randn(1,C)*e_stds[0,1]+e_mean[0,1]).repeat(C,axis=0),
                           (np.random.randn(1,C)*e_stds[1,1]+e_mean[1,1]).repeat(C,axis=0)))
                 ))
            
            JE_mean = je.repeat(c_size.flatten(),axis=0).repeat(c_size.flatten(),axis=1)
        else:
            JE_mean = EI_block_diag(e_size,e_mean)*(1-JC_mask)

        
        JE_prob = EI_block_diag(e_size,e_prob)*(1-JC_mask)
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
    def stimulation_protocol(c_range,time_st,time_en,N,n_record,stim_d,rest_d,
                             feasible,amplitude,repetition=1,fraction_stim=.8,
                             fontsize=20,visualize=True,save=False,file=None,
                             save_data=False):
        """Create random stimulation protocol for nodes of a network given 
            some input statistics
            
        Args:
            c_range (array): Array of (start,end) indices of the groups of nodes
                from which we want to select a subset to stimulate simultaneously
            time_st ()
            time_en ()
            N (integer): Total number of nodes in the network
            n_record (integer): Number of nodes that will be randomly selected
                to record from during the stimultion experiment
            stim_d (float): Duration of the stimulation
            rest_d (float): Duration of resting after each stimulation (choose 
                relative to stim_d)
            feasible (array): Boolean array determining which neurons are 
                feasible to record from
            amplitude (float): Strength of the stimulation
            repetition (integer): Number of the repetition of stimulation per 
                node
            fraction_stim (float): Fraction of nodes to be stimulated in each 
                node group
            fontsize (integer): Font size for plotting purposes
            visualize (bool): If True the resulting matrix will be plotted
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot and data
            save_data (bool): If True the generated stimulation protocol 
                information will be saved in a mat file

        Returns: I,t_stim,recorded,stimulated
            numpy.ndarray: Stimulation pattern represented as a matrix (NxT) 
                where N is the number of nodes and T is the number of time 
                points, the elements of the matrix correspond to stimulation
                strength
            numpy.ndarray: Timing in which the stimulation is sampled
            array: Feasible node indices in the network that are selected to 
                record from (based on the input 'feasible' criterion)
            array: Stimulated node indices
        """
        
        t_stim = np.linspace(time_st,time_en,repetition*int((stim_d+rest_d)/stim_d)*(len(c_range)))
        I = np.zeros((len(t_stim),N))
        
        stimulated = []
        for c in range(len(c_range)):
            sz = int(((c_range[c][1]-c_range[c][0])*fraction_stim).round())
            rand_sample = np.random.choice(np.arange(c_range[c][0],c_range[c][1]),
                   size=sz,replace=False).astype(int)
            stimulated.append(rand_sample)
            
        recorded = []
        for c in range(len(c_range)):
            rand_sample = stimulated[c]
            rand_sample = rand_sample[np.where(feasible[rand_sample])[0]]
            recorded += list(rand_sample[:n_record])
        
        
        for r in range(repetition):
            clusters = np.arange(len(c_range))
            np.random.shuffle(clusters)
            
            for c_idx,c in enumerate(clusters):
                time_idx = r*len(clusters)+c_idx
                rand_sample = stimulated[c]
                d1 = 1
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
            savemat(file+'.mat',{'c_range':c_range,'time_st':time_st,'time_en':time_en,
                                 'N':N,'n_record':n_record,'stim_d':stim_d,
                                 'rest_d':rest_d,'feasible':feasible,'amplitude':amplitude,
                                 'repetition':repetition,'fraction_stim':fraction_stim,
                                 'I':I,'t_stim':t_stim,'recorded':recorded,'stimulated':stimulated})
                
        return I,t_stim,recorded,stimulated
    
    
    @staticmethod
    def divide_clusters(c_range,C=10,C_std=.1):
        """Divide clusters into smaller groups
            
        Args:
            c_range (array): Array of (start,end) indices of the clusters
            C (integer): Number of smaller groups that we want the clusters
                to be divided to
            C_std (float): Standard deviation of the resulting subclusters

        Returns:
            array: Sub-clusters
        
        """
        c_range_ = []
        for c in c_range:
            c_ = np.round(((c[1]-c[0])/C)*C_std*np.random.randn(C)) + ((c[1]-c[0])/C).astype(int)
            c_[-1] = (c[1]-c[0]) - c_[:-1].sum()
            C_range = np.cumsum(np.hstack((c[0],c_))).astype(int)
            C_range = [(C_range[ci],C_range[ci+1]) for ci in range(len(C_range)-1)]
            c_range_ += list(C_range)
        
        return np.array(c_range_)
    
    @staticmethod
    def aggregate_spikes(spk,ind):
        """Aggregate spikes from multiple channels
            
        Args:
            spk (array): Array of (channel,spike_time)
            ind (array): Indices of the nodes that we want to aggregte their 
                spikes

        Returns:
            array: Aggregated spikes
        """
        
        if isinstance(spk[0], np.ndarray):
            return [reduce(np.union1d, tuple([spk[i].tolist() for i in ind_])) for ind_ in ind]
        else:
            return [reduce(np.union1d, tuple([spk[i] for i in ind_])) for ind_ in ind]
    
    @staticmethod
    def unsort(spk,ind=None,sample_n=3,ens_n=10,ens_ind=None,save_data=False,file=None):
        
        
        if ens_ind is None:
            ens_ind = [[]]*ens_n
            
        ens_spk = [[]]*ens_n
        for ens in range(ens_n):
            if len(ens_ind[ens])==0:
                ens_ind[ens] = [np.random.choice(ind_,size=sample_n,replace=False).astype(int) for ind_ in ind]
            ens_spk[ens] = Simulator.aggregate_spikes(spk,ens_ind[ens])
            
        if save_data:
            savemat(file+'.mat',{'ens_ind':ens_ind})
            
        return ens_ind,ens_spk
    
    @staticmethod
    def coarse_grain_matrix(J,C_size):
        """Coarse graining a matrix by averaging nodes in the blocks
            
        Args:
            J (numpy.ndarray): Matrix to be coarse grained
            C_size (array): Array of sizes to determine the blocks for coarse
                graining

        Returns:
            numpy.ndarray: Coarse grained matrix
        """
        
        c_ind = np.hstack((0,np.cumsum(C_size)))
        C_J = np.zeros((len(C_size),len(C_size)))*np.nan
        for i in range(len(C_size)):
            for j in range(len(C_size)):
                C_J[i,j] = np.nanmean(J[c_ind[i]:c_ind[i+1],:][:,c_ind[j]:c_ind[j+1]])
                
        return C_J
    
    
    @staticmethod
    def sequential_recording(X,rates,t,fov_sz,visualize=True,save=False,file=None):
        """Mask data according to a sequential recording experiment where the 
            recording FOV moves sequentially to cover the space
            
        Args:
            X (numpy.ndarray): Matrix to be coarse grained
            C_size (array): Array of sizes to determine the blocks for coarse
                graining
                X,rates,t,fov_sz,visualize=True,save=False,file=None
            X: Locations of the nodes in the network
            rates: Activities of the nodes in the network
            t: Time for the sampled rates
            fov_sz: Size of the field of view (FOV) for sequential recording
            visualize (bool): If true the graph will be visualized
            save (bool): If True the plot will be saved
            file (string): File address for saving the plot

        Returns:ens,ens_t
            numpy.ndarray: Masked rates according to the simulated sequential
                recording experiment
            array: Array of indices of the nodes in each ensemble (nodes in the same FOV)
            array: Array of timing of the nodes in the same ensemble
            
        """
        
        min_sz = X.min(0).copy()
        max_sz = X.max(0).copy()
        cur_sz = X.min(0).copy()
        
        ens = []
        
        rates_masked = rates.copy()*np.nan
        
        dir_ = 1
        
        if visualize:
            plt.subplots(figsize=((max_sz[1]-min_sz[1])/5,(max_sz[0]-min_sz[0])/5))
            plt.scatter(X[:,1],X[:,0])
        
            IND = [str(i) for i in range(X.shape[0])]
            for i, txt in enumerate(IND):
                plt.annotate(txt, (X[i,1], X[i,0]))
    
        plt.grid('on')
        
        while True:
            ind = np.logical_and.reduce([ (X[:,0]>=cur_sz[0]),
                            (X[:,1]>=cur_sz[1]),
                            (X[:,0]< cur_sz[0]+fov_sz[0]),
                            (X[:,1]< cur_sz[1]+fov_sz[1])])
            
            if len(ind) > 0:
                ens.append(np.where(ind)[0])
                
            
            if visualize:
                rectangle = plt.Rectangle((cur_sz[1],cur_sz[0]),fov_sz[1],fov_sz[0],fc=[0,0,0,0],ec='red')
                plt.gca().add_patch(rectangle)
            
            if cur_sz[1] >= max_sz[1]:
                cur_sz[1] -= fov_sz[1]
                cur_sz[0] = X[ens[-1],0].max()-2
                dir_ = -1
            elif cur_sz[1] < min_sz[1]:
                cur_sz[1] += fov_sz[1]
                cur_sz[0] = X[ens[-1],0].max()-2
                dir_ = 1
            else:
                if dir_ == 1:
                    cur_sz[1] = X[ens[-1],1].max()-2
                else:
                    cur_sz[1] = X[ens[-1],1].min()+2
                    
            if len(ens) > 1 and len(reduce(np.union1d,ens)) == X.shape[0]:
                break
        
        bins = np.linspace(min(t)-.1,max(t)+.1,len(ens))
        bin_edges = np.linspace(min(t)-.1,max(t)+.1,len(ens)+1)
        ens_t = [(bin_edges[i],bin_edges[i+1]) for i in range(len(bins))]
        for i in range(len(ens)):
            a = rates_masked[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:]
            a[:,ens[i]] = rates[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:][:,ens[i]].copy()
            rates_masked[np.where((t>=ens_t[i][0])&(t<ens_t[i][1]))[0],:] = a
        
        if visualize:
            if save:
                plt.savefig(file+'.eps',format='eps')
                plt.savefig(file+'.png',format='png')
                plt.savefig(file+'.pdf',format='pdf')
                plt.close('all')
            else:
                plt.show()
        
        return rates_masked,ens,ens_t
    
