# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: ff215, Amin
"""

import numpy as np
import networkx as nx 
from functools import partial
import matplotlib.pyplot as plt
from networkx import convert_matrix
from sklearn.decomposition import PCA
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import interp1d as intp

class Simulator(object):
    
    def __init__(self,duration=2,nSteps=10000,attractor='Rossler',parameters={'alpha':.2,'beta':.2,'gamma':5.7}):

        self.duration=duration
        self.nSteps=nSteps
        self.attractor=attractor
        self.params=parameters
        
    dim={}    
    
    def Rossler(self,t,x):
        dxdt = [-(x[1]+x[2]),
                x[0]+self.params["alpha"]*x[1],
                self.params["beta"]+x[2]*(x[0]-self.params["gamma"])]
        return dxdt
        
    dim['Rossler']=3
    
    def RosslerWithInput(self,t,x):
        interp_input = intp(self.params["input_t"],self.params["input"])
        dxdt = [-(x[1]+x[2])*(interp_input(t)),
                x[0]+self.params["alpha"]*x[1],
                self.params["beta"]+x[2]*(x[0]-self.params["gamma"])]
        return dxdt
    
    dim['RosslerWithInput']=3
    
    def Downstream(self,t,x):
        interp_input = intp(self.params["input_t"],self.params["input"])
        dxdt = -self.params["lambda"]*x + np.tanh(self.params["eta"]*x + \
                           interp_input(t))
        self.dim=1
        return dxdt
    
    dim['Downstream']=1
    
    def solve_ode(self):

        f=getattr(self, self.attractor)
        t=np.linspace(0,self.duration,self.nSteps)

        x = solve_ivp(fun=lambda t,x: f(t,x),\
                      t_span=(0,self.duration),\
                      y0=np.zeros(self.dim[self.attractor]),\
                      t_eval=t)
        return x.t,x.y.T #times = x.t ; xvalues = x.y
      

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
        
        if parameters['connectivity'] == 'Gaussian':
            J  = g*Simulator.normal_connectivity(N,g)
        elif parameters['connectivity'] == 'small_world':    
            J = g*Simulator.erdos_renyi_connectivity(N,parameters['conn_prob'])
    
    
    
        ## Input current
        
        I_zero              = np.zeros((N,T))
        I_step              = np.tile(np.concatenate((np.zeros((1, spon)), inp*np.ones((1, T - spon))),axis=1), (N,1))
        I_const             = inp*np.ones((N, T))
        
    
        if inp_t == 'zero':
            I = I_zero
        elif inp_t == 'const':
            I = I_const
        elif inp_t == 'step':
            I = I_step
    
    
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

    def visualize(self,x):

        if len(x)==1:
            plt.plot(x)
            plt.ylabel('X')
        elif len(x)==2:
            plt.plot(x[0,:],x[1,:])
            plt.xlabel('X1')
            plt.ylabel('X2')
        elif len(x)==3:
            fig=plt.figure(figsize=(12,10))
            ax=fig.gca(projection='3d')
            ax.plot(x[0,:],x[1,:],x[2,:])
            ax.set_xlabel('X axis')
            ax.set_ylabel('Y axis')
            ax.set_zlabel('Z axis')
            plt.show()
        else:
            pca = PCA(n_components=3)
            x_pca = pca.fit(np.transpose(x))
            print("original shape:   ", x.shape)
            print("transformed shape:", x_pca.shape)
            plt.fig=plt.figure(figsize=(12,10))
            ax=fig.gca(projection='3d')
            ax.plot(x_pca[0,:],x_pca[1,:],x_pca[2,:])
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC')
            ax.set_zlabel('PC3')
            plt.show()            
            # grid on; axis off; axis tight
            # daspect([1,1,1]); view(-30,-20);
            #set(gca,'color',[0.7,0.7,0.7]);

    @staticmethod
    def erdos_renyi_connectivity(N,p):
        G= nx.erdos_renyi_graph(N,p) 
#        nx.draw(G, with_labels=True)
#        plt.show() 
        return convert_matrix.to_numpy_array(G)
    
    @staticmethod
    def normal_connectivity(N,g):
        return g*np.random.normal(loc=0.0, scale=1/N, size=(N,N))
    
    
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