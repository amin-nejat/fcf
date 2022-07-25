# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: Amin
"""

from simulator import connectivity as cnn

from scipy.integrate import solve_ivp

import numpy as np
import pickle

# %%
class Nonlinear:
    def __init__(self,D,pm,discrete=False,B=None):
        self.pm = pm
        self.B = np.eye(D) if B is None else np.array(B)
        self.discrete = discrete
        if 't_eval' not in self.pm.keys(): self.pm['t_eval'] = None
        
    def run(self,T,u=None,dt=.1,x0=np.random.randn(1,3)):
        step_ = lambda t,x: self.step(t,x)
            
        t = np.arange(0,T,dt)
        if type(x0) is not np.ndarray: x0 = np.array(x0)
        if self.discrete: x = np.zeros((len(t),x0.shape[0],x0.shape[1]))
                
        if self.discrete:
            x[0,:,:] = x0
            for i in range(1,len(t)):
                if u is not None: x[i,:,:] = x[i-1,:,:] + dt*self.step(t[i],x[i-1,:,:],u=u(t[i])[None])
                else: x[i,:,:] = x[i-1,:,:] + dt*self.step(t[i],x[i-1,:,:])
        else:
            x = np.array([solve_ivp(step_,[min(t),max(t)],x0[i,:],t_eval=self.pm['t_eval']) for i in range(x0.shape[0])])
            
        return t,x
    
    def obs(self,t,x,u):
        return x
    
    def linearize(self,x):
        t = np.array([0])
        step_ = lambda x: self.step(t,x[None,:])
        J = np.autograd.functional.jacobian(step_, x, create_graph=False, strict=False)
        return J
    
    def setp(self,t,x,u=None):
        raise NotImplementedError
        
    def save(self,filename):
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    @staticmethod
    def load(filename):
        with open(filename, 'rb') as file:
            return pickle.load(file)


# %%
class Rossler(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'alpha' in keys or 'beta' in keys or 'gamma' in keys
        super(Rossler, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        dxdt = np.stack(
                    [-(x[:,1]+x[:,2]),
                    x[:,0]+self.pm['alpha']*x[:,1],
                    self.pm['beta']+x[:,2]*(x[:,0]-self.pm['gamma'])]
                ).T
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
            
        return dxdt

# %%
class Lorenz(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(Lorenz, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        dxdt = np.stack(
                [self.pm['s']*(x[:,1]-x[:,0]),
                self.pm['r']*x[:,0]-x[:,1]-x[:,0]*x[:,2],
                x[:,0]*x[:,1]-self.pm['b']*x[:,2]]
            ).T
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt
    
# %%
class RosslerDownstream(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        super(RosslerDownstream, self).__init__(D,pm,discrete=discrete,B=B)
        
        if 'J' not in keys:
            J1 = self.pm['g_i']*cnn.bipartite_connectivity(3,self.pm['N']-3,self.pm['bernoulli_p'],visualize=False)[3:,:3]
            J2 = self.pm['g_r']*cnn.normal_connectivity(self.pm['N']-3,1)
            self.pm['J'] = np.hstack((J1,J2))
            
        if 't_eval' not in keys:
            self.pm['t_eval'] = None

    def step(self,t,x,u=None):
        dxdt_u = np.stack([
                    -(x[:,1]+x[:,2]),
                    x[:,0]+self.pm['alpha']*x[:,1],
                    self.pm['beta']+x[:,2]*(x[:,0]-self.pm['gamma'])]
                ).T
        dxdt_d = -self.pm['lambda']*x[:,3:] + 10*np.tanh(np.einsum('nm,bm->bn',self.pm['J'],x))
        
        dxdt = np.concatenate((dxdt_u,dxdt_d),axis=1)
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt
    
# %%
class Thomas(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(Thomas, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        dxdt = [np.sin(x[1])-self.pm['b']*x[0],np.sin(x[2])-self.pm['b']*x[1],np.sin(x[0])-self.pm['b']*x[2]]
        return dxdt

# %%
class Langford(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(Langford, self).__init__(D,pm,discrete=discrete,B=B)

    def step(self,t,x,u=None):
        dxdt = [(x[2]-self.pm['b'])*x[0]-self.pm['d']*x[1],
                self.pm['d']*x[0]+(x[2]-self.pm['b'])*x[1],
                self.pm['c']+self.pm['a']*x[2]-x[2]**3/3-(x[0]**2+x[1]**2)*(1+self.pm['e']*x[2])+self.pm['f']*x[2]*(x[0]**3)]
        return dxdt
    
# %%
class ChaoticRate(Nonlinear):
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(ChaoticRate, self).__init__(D,pm,discrete=discrete,B=B)


    def step(self,t,x,u=None):
    
        phi = np.zeros(x.shape)
        phi[x<=0] = self.pm['R0']*np.tanh(x[x<=0]/self.pm['R0'])
        phi[x>0]  = (self.pm['Rmax']-self.pm['R0'])*np.tanh(x[x>0]/(self.pm['Rmax']-self.pm['R0']))
    
        r    = self.pm['R0']+phi
        dxdt = 1/self.pm['tau']*(-x+self.pm['J']@r)
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt

# %%
class KadmonRate():
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(KadmonRate, self).__init__(D,pm,discrete=discrete,B=B)

    def step(self,t,x,u=None):
        phi = self.pm['phi'](x)
        η = self.pm['J']@phi
        dxdt = -x + η
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt

# %%
class LucaSpiking():
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        super(LucaSpiking, self).__init__(D,pm,discrete=discrete,B=B)
    
    def step(self,t,x,u=None):
        raise NotImplementedError

# %%
class HanselSpiking():
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        super(HanselSpiking, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        raise NotImplementedError
    
