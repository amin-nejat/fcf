# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:39:40 2020

@author: Amin
"""

from simulator import connectivity as cnn

from scipy.integrate import solve_ivp
from scipy import interpolate

import numpy as np
import pickle

# %%
class RateModel:
    '''Abstract class for a rate network
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        self.D = D
        self.pm = pm
        self.discrete = discrete
        self.B = np.eye(D) if B is None else np.array(B)

        if 't_eval' not in self.pm.keys(): self.pm['t_eval'] = None
        
    def run(self,T,u=None,dt=.1,x0=None):
        if x0 is None: x0 = np.random.randn(1,self.D)
        step_ = lambda t,x: self.step(t,x)
            
        t = np.arange(0,T,dt) if self.pm['t_eval'] is None else self.pm['t_eval']
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
class SpikingModel(RateModel):
    '''Abstract class for a spiking network, which itself follows some continuous underlying rate dynamics
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        super(SpikingModel, self).__init__(D,pm,discrete,B)
        
    def run(self,T,u=None,dt=.1,x0=None):
        self.pre_run()
        
        self.spikes = []
        self.last_t = -self.pm['T']
        self.current = np.zeros((self.pm['N']))
        
        self.pm['t_eval'] = np.arange(-T,T,dt)
        
        t,x = super().run(T,u,dt,x0)
        
        spk = [[]]*self.D
        for n in range(self.D):
            spk[n] = [s[1] for s in self.spikes if s[0] == n]
            
        self.post_run()
            
        return t,x,spk,self.spikes
        
    def pre_run(self):
        pass
        
    def post_run(self):
        pass
        

# %%
class Rossler(RateModel):
    '''https://en.wikipedia.org/wiki/R%C3%B6ssler_attractor
    '''
    def __init__(self,D,pm,discrete=True,B=None):
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
class Lorenz(RateModel):
    '''https://en.wikipedia.org/wiki/Lorenz_system
    '''
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
class LDS(RateModel):
    '''Linear Dynamical System
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        keys = pm.keys()
        assert 'std' in keys
        super(LDS, self).__init__(D,pm,discrete=discrete,B=B)
        if 'J' not in keys:
            if 'M' in keys: self.pm['J'] = cnn.downstream_uniform_connectivity(self.pm['M'],self.pm['N']-self.pm['M'],self.pm['g'])
            else: self.pm['J'] = cnn.normal_connectivity(self.pm['N'],self.pm['g'])
        
    def step(self,t,x,u=None):
        dxdt = -np.einsum('nm,bm->bn',self.pm['J'],x) + np.random.randn(*tuple(x.shape))*self.pm['std']
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt
    
# %%
class RosslerDownstream(RateModel):
    '''Upstream Rossler attractor driving a downstream chaotic network
    '''
    
    def __init__(self,D,pm,discrete=True,B=None):
        keys = pm.keys()
        super(RosslerDownstream, self).__init__(D,pm,discrete=discrete,B=B)
        
        if 'J' not in keys:
            J1 = self.pm['g_i']*cnn.bipartite_connectivity(3,self.pm['N']-3,self.pm['bernoulli_p'])[3:,:3]
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
class LorenzDownstream(RateModel):
    '''Upstream Lorenz attractor driving a downstream chaotic network
    '''
    
    def __init__(self,D,pm,discrete=True,B=None):
        keys = pm.keys()
        super(LorenzDownstream, self).__init__(D,pm,discrete=discrete,B=B)
        
        if 'J' not in keys:
            J1 = self.pm['g_i']*cnn.bipartite_connectivity(3,self.pm['N']-3,self.pm['bernoulli_p'])[3:,:3]
            J2 = self.pm['g_r']*cnn.normal_connectivity(self.pm['N']-3,1)
            self.pm['J'] = np.hstack((J1,J2))
            
        if 't_eval' not in keys:
            self.pm['t_eval'] = None

    def step(self,t,x,u=None):
        dxdt_u = np.stack(
                [self.pm['s']*(x[:,1]-x[:,0]),
                self.pm['r']*x[:,0]-x[:,1]-x[:,0]*x[:,2],
                x[:,0]*x[:,1]-self.pm['b']*x[:,2]]
            ).T
        
        dxdt_d = -self.pm['lambda']*x[:,3:] + 10*np.tanh(np.einsum('nm,bm->bn',self.pm['J'],x))
        
        dxdt = np.concatenate((dxdt_u,dxdt_d),axis=1)
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt
    
# %%
class Downstream(RateModel):
    '''A downstream network driven with inputs provided externally
    '''
    def __init__(self,D,pm,B=None):
        keys = pm.keys()
        super(Downstream, self).__init__(D,pm,B=B)
       
        noise = np.random.randn(self.pm['I'].shape[0],self.pm['I'].shape[1])*self.pm['noise_std']
        if 'U_J' not in keys:
            self.pm['U_J'] = cnn.bipartite_connectivity(self.pm['I'].shape[1],self.pm['N'],self.pm['bernoulli_p'])[self.pm['I'].shape[1]:,:self.pm['I'].shape[1]]
            
        if 'J' not in keys:
            self.pm['J'] = cnn.normal_connectivity(self.pm['N'],1)
        self.upstream = interpolate.interp1d(
                self.pm['t_eval'],
                self.pm['U_J']@(self.pm['I'].T+noise.T),
                kind='linear',
                bounds_error=False
            )
        
    def step(self,t,x,u=None):
        I = np.einsum('mn,bm->bn',self.B,u) if u is not None else 0
        dxdt= -self.pm['lambda']*x + 10*np.tanh(self.pm['g_r']*np.einsum('nm,bm->bn',self.pm['J'],x)+self.pm['g_i']*self.upstream(t)+I)
        return dxdt
    
    


# %%
class Thomas(RateModel):
    '''https://en.wikipedia.org/wiki/Thomas%27_cyclically_symmetric_attractor
    '''
    def __init__(self,D,pm,discrete=False,B=None):
        keys = pm.keys()
        assert 'r' in keys or 'b' in keys or 's' in keys
        super(Thomas, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        dxdt = [np.sin(x[1])-self.pm['b']*x[0],np.sin(x[2])-self.pm['b']*x[1],np.sin(x[0])-self.pm['b']*x[2]]
        return dxdt

# %%
class Langford(RateModel):
    '''https://link.springer.com/chapter/10.1007/978-3-0348-6256-1_19
    '''
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
class DirectedAcyclicRate(RateModel):
    '''Simple rate model with directed acyclic connectivity, no recurrence
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        super(DirectedAcyclicRate, self).__init__(D,pm,discrete=discrete,B=B)
        self.pm['J'] = cnn.dag_connectivity(pm['N'],pm['p'],pm['g'])

    def step(self,t,x,u=None):
        dxdt = 1/self.pm['tau']*(-np.einsum('nm,bm->bn',self.pm['J'],x))
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt
    
# %%
class ChaoticRate(RateModel):
    '''Simple rate model with tanh nonlinearity
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        keys = pm.keys()
        super(ChaoticRate, self).__init__(D,pm,discrete=discrete,B=B)
        
        if 'J' not in keys:
            if 'M' in keys: self.pm['J'] = cnn.downstream_uniform_connectivity(self.pm['M'],self.pm['N']-self.pm['M'],self.pm['g'])
            else: self.pm['J'] = cnn.normal_connectivity(self.pm['N'],self.pm['g'])
        
        
    def step(self,t,x,u=None):
        phi = np.zeros(x.shape)
        phi[x<=0] = self.pm['R0']*np.tanh(x[x<=0]/self.pm['R0'])
        phi[x>0] = (self.pm['Rmax']-self.pm['R0'])*np.tanh(x[x>0]/(self.pm['Rmax']-self.pm['R0']))
    
        r = self.pm['R0']+phi
        dxdt = 1/self.pm['tau']*(-x+np.einsum('nm,bm->bn',self.pm['J'],r)) + self.pm['baseline']
        
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)
        
        return dxdt

# %%
class KadmonRate(RateModel):
    '''Network studied by J. Kadmon
    '''
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
class ClusteredSpiking(SpikingModel):
    '''Spiking network with clustered E-I connectivity exhibiting switching dynamics
    '''
    def __init__(self,D,pm,discrete=True,B=None):
        keys = pm.keys()
        super(ClusteredSpiking, self).__init__(D,pm,discrete=discrete,B=B)

        if 'J' not in keys:
            self.pm['J'], self.pm['cluster_size'] = cnn.clustered_connectivity(
                        N=self.pm['N'],
                        EI_frac=self.pm['EI_frac'],
                        C=self.pm['C'],
                        C_std=self.pm['C_std'],
                        clusters_mean=self.pm['clusters_mean'],
                        clusters_stds=self.pm['clusters_stds'],
                        clusters_prob=self.pm['clusters_prob'],
                        external_mean=self.pm['external_mean'],
                        external_stds=self.pm['external_stds'],
                        external_prob=self.pm['external_prob']
                    )
            
    
    def step(self,t,x,u=None):
        self.refr = np.maximum(-0.001,self.refr-(t-self.last_t))
        x[0,self.refr > 0] = self.pm['v_rest'][self.refr > 0]
        
        fired = (x[0,:] >= self.pm['theta'])
        x[0,fired] = self.pm['v_rest'][fired]
        
        [self.spikes.append((s,t)) for s in np.where(fired)[0]]
        self.current *= np.exp((t-self.last_t)*self.pm['f_mul'])
        self.current[fired] += self.pm['f_add'][fired]
        self.refr[fired] = self.pm['tau_arp']
        
        dxdt = -x[0,:]/self.pm['tau_m'] + self.pm['J']@self.current + self.pm['baseline']
        if u is not None: dxdt += np.einsum('mn,bm->bn',self.B,u)[:,0]
        
        self.last_t = t
        
        dxdt[fired] = 0
        return dxdt[None,:]
    
    def pre_run(self):
        self.refr = np.zeros((self.pm['N']))
        

    

# %%
class HanselSpiking(SpikingModel):
    '''Spiking network studied by D. Hansel
    '''
    def __init__(self,D,pm,discrete=False,B=None):
        super(HanselSpiking, self).__init__(D,pm,discrete=discrete,B=B)
        
    def step(self,t,x,u=None):
        [self.spikes.append((spk,t)) for spk in np.where(x >= self.pm['theta'])[0]]
        self.current *= np.exp(self.last_t-t)
        self.current[x >= self.pm['theta']] += \
                        (1/(self.pm['tau1']-self.pm['tau2']))* \
                        (np.exp(-t/self.pm['tau1']) - np.exp(-t/self.pm['tau2']))
        
        x[x >= self.pm['theta']] = self.pm['v_rest']
        self.I_syn = -(self.pm['I_syn_avg']/self.pm['N'])*self.pm['J']@self.current
        dxdt = (1/self.pm['C'])*(-self.pm['g_l']*(x-self.pm['v_rest'])+self.I_syn)
        if u is not None: dxdt += (1/self.pm['C'])*np.einsum('mn,bm->bn',self.B,u)
        
        self.last_t = t
        
        return dxdt
    
