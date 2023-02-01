# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 21:24:16 2022

@author: Amin
"""

from causality import helpers as inth
from simulator import networks as net
from simulator import helpers as simh

from scipy.io import loadmat
import numpy as np

import os

# %%
class RateDataset:
    def __init__(self,pm,load=False,save=False,file=None):
        self.network = eval('net.'+pm['model'])(pm['N'],pm=pm)
        self.recorded = np.arange(pm['N'])
        
        self.save = save
        self.load = load
        self.file = file
        
    def load_rest(self,pm):
        if self.load:
            result = np.load(self.file+'rest.npy',allow_pickle=True).item()
            self.mask = result['mask']
            t,y = result['t'],result['y']
            self.stimulated_recorded = result['stimulated_recorded']

        else:
            t,y = self.network.run(pm['T'],dt=pm['dt'],x0=.5*np.random.randn(1,pm['N'])/pm['N'])
            I,t_stim,_,stimulated,u = inth.stimulation_protocol(
                        [(i,i+1) for i in self.recorded],
                        time_st=0,
                        time_en=pm['T_stim'],
                        N=pm['N'],
                        n_record=pm['n_record'],
                        stim_d=pm['stim_d'],
                        rest_d=pm['rest_d'],
                        feasible=np.ones(pm['N']).astype(bool),
                        amplitude=pm['amplitude_c']*np.ones(pm['N']),
                        repetition=pm['repetition'],
                        fraction_stim=pm['fraction_stim']
                    )
            self.I = I
            self.u = u
            self.t_stim = t_stim
            self.stimulated_recorded = stimulated
        
        self.mask = np.ones((pm['recorded'],pm['recorded'])).astype(bool)
        self.mask[:,self.stimulated_recorded] = False
        np.fill_diagonal(self.mask, True)
        
        if self.save:
            np.save(self.file+'rest',{
                    'y':y,'t':t,'mask':self.mask,'stimulated_recorded':self.stimulated_recorded
                })
        
        return y[:,0,self.recorded],t,{}
    
    def load_stim(self,pm):
        if self.load:
            result = np.load(self.file+'stim.npy',allow_pickle=True).item()
            self.mask = result['mask']
            self.I = result['I']
            self.t_stim = result['t_stim']
            t,y = result['t'],result['y']
        else:
            self.network.pm['I_J'] = np.eye(pm['N'])
            t,y = self.network.run(pm['T_stim'],dt=pm['dt'],x0=np.random.randn(1,pm['N']),u=self.u)
        
        out = {'I':self.I,'t_stim':self.t_stim}
        
        if self.save:
            np.save(self.file+'stim',{
                    'y':y,'t':t,
                    'mask':self.mask,
                    'I':self.I,'t_stim':self.t_stim,
                })
        
        return y[:,0,self.recorded],t,out

# %%
class RosslerDownstreamDataset:
    def __init__(self,pm,load=False,save=False,file=None):
        self.network = net.RosslerDownstream(pm['N'], pm)
        self.recorded = np.arange(10)
        
        self.save = save
        self.load = load
        self.file = file
        
    def load_rest(self,pm):
        if self.load:
            result = np.load(self.file+'rest.npy',allow_pickle=True).item()
            self.mask = result['mask']
            t,y = result['t'],result['y']
            self.stimulated_recorded = result['stimulated_recorded']

        else:
            t,y = self.network.run(pm['T'],dt=pm['dt'],x0=np.random.randn(1,pm['N']))
        
            I,t_stim,_,stimulated,u = inth.stimulation_protocol(
                        [(i,i+1) for i in self.recorded],
                        time_st=0,
                        time_en=pm['T_stim'],
                        N=pm['N'],
                        n_record=pm['n_record'],
                        stim_d=pm['stim_d'],
                        rest_d=pm['rest_d'],
                        feasible=np.ones(pm['N']).astype(bool),
                        amplitude=pm['amplitude_c']*np.ones(pm['N']),
                        repetition=pm['repetition'],
                        fraction_stim=pm['fraction_stim']
                    )
            self.I = I
            self.u = u
            self.t_stim = t_stim
            self.stimulated_recorded = stimulated
        
        self.mask = np.ones((pm['recorded'],pm['recorded'])).astype(bool)
        self.mask[:,self.stimulated_recorded] = False
        np.fill_diagonal(self.mask, True)
        self.mask[:3,:3] = True
        
        if self.save:
            np.save(self.file+'rest',{
                    'y':y,'t':t,'mask':self.mask,'stimulated_recorded':self.stimulated_recorded
                })
        
        return y[:,0,self.recorded],t,{}
    
    def load_stim(self,pm):
        if self.load:
            result = np.load(self.file+'stim.npy',allow_pickle=True).item()
            self.mask = result['mask']
            self.I = result['I']
            self.t_stim = result['t_stim']
            t,y = result['t'],result['y']
        else:
            self.network.pm['I_J'] = np.eye(pm['N'])
            t,y = self.network.run(pm['T_stim'],dt=pm['dt'],x0=np.random.randn(1,pm['N']),u=self.u)
            
        out = {'I':self.I,'t_stim':self.t_stim}
        
        if self.save:
            np.save(self.file+'stim',{
                    'y':y,'t':t,
                    'mask':self.mask,
                    'I':self.I,'t_stim':self.t_stim
                })
        
        return y[:,0,self.recorded],t,out


# %%
class ClusteredSpikingDataset:
    def __init__(self,pm,load=False,save=False,file=None):
        E = round(pm['N']*pm['EI_frac'])
        I = round(pm['N']*(1-pm['EI_frac']))

        pm['theta'] = np.concatenate((np.ones((E))*pm['theta_c'][0],np.ones((I))*pm['theta_c'][1]))
        pm['v_rest'] = np.concatenate((np.zeros((E))*pm['v_rest_c'][0],np.zeros((I))*pm['v_rest_c'][1]))
        pm['tau_syn'] = np.concatenate((np.ones((E))*pm['tau_syn_c'][0],np.ones((I))*pm['tau_syn_c'][1]))
        pm['tau_m'] = np.concatenate((np.ones((E))*pm['tau_m_c'][0],np.ones((I))*pm['tau_m_c'][0]))
        pm['f_mul'] = -1/pm['tau_syn']
        pm['f_add'] = 1/pm['tau_syn']
        factor = ((1000/pm['N'])**(1/2))*5*0.8*E*.2
        pm['baseline'] = np.concatenate((
                factor*pm['baseline_c'][0][0]*(np.ones((E))+(pm['baseline_c'][0][1])*(2*np.random.rand(E)-1)),
                factor*pm['baseline_c'][1][0]*(np.ones((I))+(pm['baseline_c'][1][1])*(2*np.random.rand(I)-1))
            ))

        self.network = net.ClusteredSpiking(pm['N'], pm)
        
        self.save = save
        self.load = load
        self.file = file
    
    def load_rest(self,pm):
        if self.load:
            result = np.load(self.file+'rest.npy',allow_pickle=True).item()
            self.mask = result['mask']
            t,x,spikes,spikes_flat = result['t'],result['x'],result['spikes'],result['spikes_flat']
            self.stimulated_recorded = result['stimulated_recorded']
        else:
            t,x,spikes,spikes_flat = self.network.run(pm['T'],dt=pm['dt'])
            cluster_starts = np.hstack((0,np.cumsum(self.network.pm['cluster_size'].flatten())))
    
            feasible_clusters = np.array([
                    np.mean([len(spikes[i]) for i in range(cluster_starts[c],cluster_starts[c+1])]) 
                    for c in range(len(cluster_starts)-1)
                ]) > pm['min_firing_rate']
            
            cluster_intervals = [(cluster_starts[c],cluster_starts[c+1]) for c in range(pm['C']) if feasible_clusters[:pm['C']][c]]
            
            feasible = np.array([len(spikes[i]) for i in range(len(spikes))])/np.ptp(t)>pm['min_firing_rate']
    
            I,t_stim,recorded,stimulated,u = inth.stimulation_protocol(
                        cluster_intervals,
                        time_st=-pm['T_stim'],
                        time_en=pm['T_stim'],
                        N=pm['N'],
                        n_record=pm['per_cluster'],
                        stim_d=pm['stim_d'],
                        rest_d=pm['rest_d'],
                        feasible=feasible,
                        amplitude=pm['amplitude_c']*pm['baseline'],
                        repetition=pm['repetition'],
                        fraction_stim=pm['fraction_stim'],
                    )
            
            
            self.stimulated = np.unique([channel for cluster_stimulated in stimulated for channel in cluster_stimulated])
            self.recorded = np.unique(recorded)
            
            self.stimulated_recorded = [np.where(self.recorded == i)[0][0]  for i in self.stimulated if len(np.where(self.recorded == i)[0])>0]
            self.u = u
            
            self.I = I
            self.t_stim = t_stim
            
        self.mask = np.ones((len(self.recorded),len(self.recorded))).astype(bool)
        self.mask[:,self.stimulated_recorded] = False
        np.fill_diagonal(self.mask, True)
        
        spk = [np.array(spikes[i]) for i in self.recorded]
        rates,t_rates = simh.spktimes_to_rates(
                spk,
                n_bins=int(pm['spktimes_to_rates_ptp']*np.ptp(t)/pm['dt']),
                rng=(min(t),max(t)),
                sigma=pm['spktimes_to_rates_sigma'],
                method='gaussian',
            )
        
        out = {
            't':t,
            'x':x[:,0,self.recorded],
            'spikes':spikes,
            'spikes_flat':spikes_flat
           }
        
        if self.save:
            np.save(self.file+'rest',{
                    'x':x,'t':t,'mask':self.mask,
                    'spikes':spikes,'spikes_flat':spikes_flat,
                    'stimulated_recorded':self.stimulated_recorded
                })
        
        return rates,t_rates,out
    
    def load_stim(self,pm):
        if self.load:
            result = np.load(self.file+'stim.npy',allow_pickle=True).item()
            self.mask = result['mask']
            self.I = result['I']
            self.t_stim = result['t_stim']
            t,x,spikes,spikes_flat = result['t'],result['x'],result['spikes'],result['spikes_flat']
        else:
            t,x,spikes,spikes_flat = self.network.run(pm['T_stim'],dt=pm['dt'],u=self.u)
            spk = [np.array(spikes[i]) for i in self.recorded]
        
        rates,t_rates = simh.spktimes_to_rates(
                spk,
                n_bins=int(pm['spktimes_to_rates_ptp']*np.ptp(t)/pm['dt']),
                rng=(min(t),max(t)),
                sigma=pm['spktimes_to_rates_sigma'],
                method='gaussian',
            )
        
        out = {
            't':t,
            'x':x[:,0,self.recorded],
            'spikes':spikes,
            'spikes_flat':spikes_flat,
            'I':self.I,
            't_stim':self.t_stim
           }
        
        if self.save:
            np.save(self.file+'rest',{
                    'x':x,'t':t,'mask':self.mask,
                    'spikes':spikes,'spikes_flat':spikes_flat,
                    'I':self.I,'t_stim':self.t_stim
                })
        
        return rates,t_rates,out
    

# %%
class RoozbehLabDataset:
    def __init__(self,pm,load=False,save=False,file=None):
        self.dict_stim = RoozbehLabDataset.load_spiking_data(pm['stim_file'])
        self.dict_rest = RoozbehLabDataset.load_spiking_data(pm['rest_file'])

        self.stimulated_recorded = np.unique([
                self.dict_stim['stim_info'][i][0] 
                for i in range(len(self.dict_stim['stim_info']))
            ])
        
        self.mask = np.ones((96,96)).astype(bool)
        self.mask[:,self.stimulated_recorded] = False
        np.fill_diagonal(self.mask, True)
        
        self.layout = RoozbehLabDataset.array_maps()[os.path.split(pm['stim_file'])[1][0]]
        
        self.save = save
        self.load = load
        self.file = file
        
    @staticmethod
    def load_spiking_data(file):
         FIRA = loadmat(file)['FIRA'][0].tolist()
    
         if(FIRA[1].shape[0]!=FIRA[2].shape[0]):
              print('warning: number of trials differs in 2nd and 3rd FIRA field')
              
         n_trials = min(FIRA[1].shape[0],FIRA[2].shape[0])
         n_events = FIRA[1].shape[1]
         max_channel = 96
         
         # tuple, but to be converted into a dictionary
         events = FIRA[0][0][0][6][0].tolist()
         assert(len(events)==n_events)
         
         for event_ind in range(n_events): 
             events[event_ind]=str(events[event_ind][0])
         
         spiking_times,values = [],[]
         
         for trial in range(n_trials):
              spiking_times.append([])
              values.append([])
         
              for event in range(n_events):
                   temp = FIRA[1][trial][event].tolist()
                   
                   while type(temp) == list and len(temp) == 1:
                        if type(temp[0]) == list: temp = temp[0]
                        else: break
                    
                   values[trial].append(temp)
                   
              for channel in range(max_channel):
                   channel_spikes = np.array([])
                   for unit_class in range(len(FIRA[2][trial,0][channel])):
                        channel_spikes = np.concatenate((channel_spikes,FIRA[2][trial,0][channel,unit_class].flatten()))
                   spiking_times[trial].append(np.sort(channel_spikes))
                        
                   
         def get_indices(event_string):
              ind = [i for i,x in enumerate(events) if x == event_string]
              return(ind[0])
         
         spk_session = [[]]*max_channel
         stim_times,stim_durations,stim_chan = [],[],[]
         
         for trl_i in range(n_trials):
             trl_offset = values[trl_i][get_indices('abs_ref')][0]-values[0][get_indices('abs_ref')][0] - values[0][get_indices('start_tri')][0]
             
             for ch_i in range (max_channel):
                  spk_session[ch_i] = np.concatenate((spk_session[ch_i],spiking_times[trl_i][ch_i]+trl_offset))
         
             if values[trl_i][get_indices('elestim')]==[1]:
                  stim_times.append(values[trl_i][get_indices('elestim_on')][0]+trl_offset)
                  stim_durations.append(values[trl_i][get_indices('elestim_off')][0]-values[trl_i][get_indices('elestim_on')][0])
                  
                  # warning: this gives the id of stimulated channels under the convention taht numbers channels from one, not zero 
                  stim_chan.append(values[trl_i][get_indices('ustim_chan')][0]-1) 
         
         assert(max_channel==len(spk_session))      
    
         data_dict = {}
         data_dict['spikes'] = [list(spk_session[ch_i]) for ch_i in range(max_channel)]
         
         spike_timings,spike_nodes = np.array([]),np.array([])
         
         for ch_i in range(max_channel):
              spike_timings=np.concatenate((spike_timings,spk_session[ch_i]))
              spike_nodes=np.concatenate((spike_nodes,ch_i*np.ones(len(spk_session[ch_i]))))
              
         assert(len(spike_timings)==len(spike_nodes))
         inds = np.argsort(spike_timings)
         spikes_flat = []
         for spike_i in range(len(spike_timings)):
              spikes_flat.append((spike_nodes[inds[spike_i]],spike_timings[inds[spike_i]]))
         data_dict['spikes_flat'] = spikes_flat
         data_dict['stim_info'] = [
                 (stim_chan[stim_i],stim_times[stim_i],stim_times[stim_i]+stim_durations[stim_i]) 
                 for stim_i in range(len(stim_times))
             ]
         
         return(data_dict)
    
    # %%
    @staticmethod
    def array_maps():
        array_map={}
        
        array_map["G"]=np.array([[3,2,1,0,4,6,8,0,14,10],\
                               [65,66,33,34,7,9,11,12,16,18],\
                               [67,68,35,36,5,17,13,23,20,22],\
                               [69,70,37,38,48,15,19,25,27,24],\
                               [71,72,39,40,42,50,54,21,29,26],\
                               [73,74,41,43,44,46,52,62,31,28],\
                               [75,76,45,47,51,56,58,60,64,30],\
                               [77,78,82,49,53,55,57,59,61,32],\
                               [79,80,84,86,87,89,91,94,63,95],\
                               [0,81,83,85,88,90,92,93,96,0]])
             
        array_map["N"]=np.array([[2,0,1,3,4,6,8,10,14,0],\
                               [65,66,33,34,7,9,11,12,16,18],\
                               [67,68,35,36,5,17,13,23,20,22],\
                               [69,70,37,38,48,15,19,25,27,24],\
                               [71,72,39,40,42,50,54,21,29,26],\
                               [73,74,41,43,44,46,52,62,31,28],\
                               [75,76,45,47,51,56,58,60,64,30],\
                               [77,78,82,49,53,55,57,59,61,32],\
                               [79,80,84,86,87,89,91,94,63,0],\
                               [0,81,83,85,88,90,92,93,96,95]])
        
        # mask 0 chanenls
        for key in array_map.keys():
             array_map[key] = np.ma.array(array_map[key],mask=False)
             empty_site = np.where(array_map[key]==0)
             for empty_ind in range(4):
                      array_map[key].mask[empty_site[0][empty_ind],empty_site[1][empty_ind]] = True          
        
        # move to Python indexing, with the lowest channel index now being 0
        for key in array_map.keys(): array_map[key] -= 1
             
        return array_map
                       
             
    def load_rest(self,pm):
        min_t = min([min(self.dict_rest['spikes'][i]) for i in range(len(self.dict_rest['spikes']))])
        max_t = max([max(self.dict_rest['spikes'][i]) for i in range(len(self.dict_rest['spikes']))])

        # t is in miliseconds => every bin is 60 ms
        n_bins = int(np.round((max_t-min_t)/pm['fs'])) 

        rates,time = simh.spktimes_to_rates(
                [np.array(spk) for spk in self.dict_rest['spikes']],
                n_bins=n_bins,
                rng=(min_t,max_t),
                method='counts'
            )
        out = {
                'spikes': self.dict_rest['spikes'],
                'spikes_flat': self.dict_rest['spikes_flat']
            }
        
        return rates,time,out
    
    def load_stim(self,pm):
        min_t = min([min(self.dict_stim['spikes'][i]) for i in range(len(self.dict_stim['spikes']))])
        max_t = max([max(self.dict_stim['spikes'][i]) for i in range(len(self.dict_stim['spikes']))])
        
        # t is in miliseconds => every bin is 60 ms
        n_bins = int(np.round((max_t-min_t)/pm['fs'])) 
        
        rates,time= simh.spktimes_to_rates(
                [np.array(spk) for spk in self.dict_stim['spikes']],
                n_bins=n_bins,rng=(min_t,max_t),method='counts'
            )
        
        out = {
                'spikes': self.dict_stim['spikes'],
                'spikes_flat': self.dict_stim['spikes_flat'],
                'stim_info': self.dict_stim['stim_info']
            }
        
        return rates,time,out

class HarveyHolographic:
    def __init__(self,pm,load=False,save=False,file=None):
        self.stim_index = np.load(pm['stim_index'])
        self.stim_index = [(a[0],a[1],a[2]) for a in self.stim_index]
        self.spikes = np.load(pm['spikes'])
        self.mask = np.zeros((self.spikes.shape[0], self.spikes.shape[0])).astype(bool)

    def load_stim(self, pm):
        return self.spikes.T, np.arange(0, self.spikes.shape[1], 1), {'stim_info' : self.stim_index}

    def load_rest(self, pm):
        return self.spikes.T, np.arange(0, self.spikes.shape[1], 1), {}
