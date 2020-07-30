# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:42:55 2020

"""


import numpy as np
import pickle
import itertools as it

whichDataset=2
whichAfferentInd=0

dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'              
dataKeys = pickle.load(open(keysLocation, "rb"))

assert(dataKeys[whichDataset][1]!=0)

stim_filename=dataFolder+'spikeData'+dataKeys[whichDataset][1]+'.p'
stimulated=pickle.load(open(stim_filename, "rb"))
spkTimes=stimulated['spk_session']
pulseStarts=np.array(stimulated['stim_times'])
pulseDurations=np.array(stimulated['stim_durations'])
stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the convention taht numbers channels from one, not zero 

stimCh=15
afferent=15
ch_inds=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(np.int64)

binSize=5
preCushion=10
postCushion=4
postInterval=750
preInterval=100


nChannels=len(spkTimes)
assert(stimCh>=1 and stimCh<=nChannels) #stimCh is the ID of the perturbed channel under the assumption that channels are numbered from one, not zero
assert(len(pulseStarts)==len(pulseDurations))
nEvents=len(pulseStarts)
   
preBinEdges=np.array([-preInterval-preCushion,-preCushion])
postBinEdges=np.arange(postCushion,postCushion+postInterval,binSize)

n_postBins=len(postBinEdges)-1
 
preCount=np.ma.array(np.zeros((nEvents,nChannels)),mask=False)
preCount.mask[:,stimCh-1]=True

postCounts=np.ma.array(np.zeros((n_postBins,nEvents,nChannels,)),mask=False)
postCounts.mask[:,:,stimCh-1]=True

# Also to initialize: meanPostRates, meanPreRates, postCounts, postIncrements

event=0

for channel in (x for x in range(nChannels) if x != stimCh-1):
     
     firstIndex=np.where(spkTimes[channel]>=pulseStarts[event]-preCushion-preInterval)[0][0]
     lastIndex=np.where(spkTimes[channel]<pulseStarts[event]+pulseDurations[event]+postCushion+postInterval)[0][-1]
     times=spkTimes[channel][firstIndex:lastIndex+1]-pulseStarts[event]
     
     preCount[event,channel]=np.histogram(times,preBinEdges)[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
     postCounts[:,event,channel]=np.histogram(times,pulseDurations[event]+postBinEdges)[0]
     
incrementsByBin=postCounts/binSize-preCount/preInterval #summed through broadcasting 
mean_incrBbyBin=np.mean(incrementsByBin,0)
std_incrByBin=np.std(incrementsByBin,0)

incrementsByLapse=np.cumsum(postCounts,0) #1
incrementsByLapse=np.transpose(incrementsByLapse, (1, 2, 0)) #2
incrementsByLapse=incrementsByLapse/np.diff(postBinEdges) #3
incrementsByLapse=np.transpose(incrementsByLapse, (2,0,1)) #4     
incrementsByLapse=incrementsByLapse-preCount/preInterval

mean_incrByLapse=np.mean(incrementsByLapse,0)
std_incrByLapse=np.std(incrementsByLapse,0)


