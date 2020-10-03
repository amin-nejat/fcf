# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:02:14 2020
 
"""


from scipy.io import loadmat
import numpy as np
import pickle 

#loads and pickle any given dataset

def convertData(FIRA):

     ##################### PYTHONIZE IT ############################
     
     print("pythonizing the dataset...")
     
     if(FIRA[1].shape[0]!=FIRA[2].shape[0]):
          print('warning: number of trials differs in 2nd and 3rd FIRA field')
     nTrials=min(FIRA[1].shape[0],FIRA[2].shape[0])
     nEvents=FIRA[1].shape[1]
     #anyTrialInd=nTrials-1 # or any number between 0 and nTrials-1
     #nChannels_total =FIRA[2][anyTrialInd,0].shape[0]  #no, must count only until CH_MAX
     CH_MAX = 96; #Number of channels
     
     events=FIRA[0][0][0][6][0].tolist() #tuple, but to be converted into a dictionary
     assert(len(events)==nEvents)
     for eventInd in range(nEvents):
          events[eventInd]=str(events[eventInd][0])
     
     spikingTimes=[]
     values=[]
     
     for trial in range(nTrials):
     
          spikingTimes.append([])
          values.append([])
     
          for event in range(nEvents):
               temp=FIRA[1][trial][event].tolist()
               while type(temp)==list and len(temp)==1:
                    if type(temp[0])==list:
                         temp=temp[0]
                    else:
                         break
               values[trial].append(temp)
               
          for channel in range(CH_MAX):
               spikingTimes[trial].append(FIRA[2][trial,0][channel,0].flatten())
     
     ##################### STITCH TOGETHER THE FAKE TRIALS #####################
               
     print("stitching trials...")
     
     # method to get index of trial events
     def IND(eventString):
          ind= [i for i,x in enumerate(events) if x == eventString]
          return(ind[0])
     
     spk_session = [[] for i in range(CH_MAX)] # an empty list of list of length CH_MAX
     stim_times=[]
     stim_durations=[]
     stim_chan=[]
     
     for trl_i in range(nTrials):
     
         #First I calculate the offset for each trial.
         #The offset is calculated as the difference of the refrence
         #time of each trial with respect to the first trial.
         #The refrence point here is the onset of fixation point so I'm
         #also adding the fixation point onset of the first trial to
         #align everything to the beginning of the first trial
     
         trl_offset = values[trl_i][IND('abs_ref')][0]-values[0][IND('abs_ref')][0] - values[0][IND('start_tri')][0]
         
         for ch_i in range (CH_MAX):
              spk_session[ch_i] = np.concatenate((spk_session[ch_i],spikingTimes[trl_i][ch_i]+trl_offset))
     
         if values[trl_i][IND('elestim')]==[1]:
              stim_times.append(values[trl_i][IND('elestim_on')][0]+trl_offset)
              stim_durations.append(values[trl_i][IND('elestim_off')][0]-values[trl_i][IND('elestim_on')][0])
              stim_chan.append(values[trl_i][IND('ustim_chan')][0]-1) #warning: this gives the id of stimulated channels under the convention taht numbers channels from one, not zero 
     
     assert(CH_MAX==len(spk_session))      

     # %% SAVE INTO DICTIONARY 
     
     dataDict= {}

     print("storing spikes...")
     # save the "spikes" entry 
     #
     dataDict["spikes"] = [list(spk_session[ch_i]) for ch_i in range(CH_MAX)]
     
     # save the "spikes_flat" entry 
     #
     print("flattening the spikes list...")
     spike_timings=np.array([])
     spike_nodes=np.array([])
     for ch_i in range(CH_MAX):
          spike_timings=np.concatenate((spike_timings,spk_session[ch_i]))
          spike_nodes=np.concatenate((spike_nodes,ch_i*np.ones(len(spk_session[ch_i]))))
     assert(len(spike_timings)==len(spike_nodes))
     nSpikes=len(spike_timings)
     print("sorting spikes by time...")
     inds=np.argsort(spike_timings)
     print("saving spikes_flat ...")
     spikes_flat=[]
     for spike_i in range(nSpikes):
          spikes_flat.append((spike_nodes[inds[spike_i]],spike_timings[inds[spike_i]]))
     dataDict["spikes_flat"]=spikes_flat
     
     # save the "stimulationArray"
     #
     print("storing stimulation array...")
     nStims=len(stim_times)
     dataDict["stim_array"] = [(stim_chan[stim_i],stim_times[stim_i],stim_times[stim_i]+stim_durations[stim_i]) for stim_i in range(nStims)]
     
     return(dataDict)
     
if __name__=="__main__":
     
     #loading all the datasets

     sourceFolder='../../../SALEH/unsorted/' #or any folder with the content of https://drive.google.com/drive/u/0/folders/12usjkXjnhjhiiRUQAjzM4jnbcqyio8s1
     targetFolder='../../DATA/' #or any existing folder where you want to store the output
     keysLocation='../../DATA/dataKeys'              
     dataKeys = pickle.load(open(keysLocation, "rb"))
     dataTypes=["resting","stimulated"]
     for dataIndex in range(len(dataKeys)):
          print('converting dataset '+str(dataIndex+1)+' of '+str(len(dataKeys))+'...')
          for dataType in range(2):
               if dataKeys[dataIndex][dataType]!=0:
                    datasetName=dataKeys[dataIndex][dataType]
                    sourceFile=sourceFolder+datasetName+'.mat'
                    FIRA= loadmat(sourceFile)['FIRA'][0].tolist()
                    dataDict=convertData(FIRA)
                    dataDict["dataset_ID"]=datasetName
                    dataDict["data_type"]=dataTypes[dataType]
                    targetFile=targetFolder+'session'+str(dataIndex)+"_part"+str(dataType+1)+'.p'
                    pickle.dump(dataDict, open(targetFile, "wb" ))
                    # dataDict = pickle.load(open(targetFile, "rb"))
                   
     