 # -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:24:16 2020
"""

import pickle
import numpy as np
from DataTools import utilities as util
import itertools as it
from scipy import stats
from Causality.responseAnalysis import analyzeResponse,causalityVsResponse
from DelayEmbedding import DelayEmbedding as DE 


dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'              
dataKeys = pickle.load(open(keysLocation, "rb"))
binSize=50

usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

dkInd=0
dk = usableDataKeys[dkInd]

resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
resting=pickle.load(open(resting_filename, "rb"))
nChannels=len(resting['spk_session'])
spk_resting=resting['spk_session']
rates_resting=util.spk2rates(spk_resting,binSize=binSize)[0] #output is a numpy array

stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
stimulated=pickle.load(open(stim_filename, "rb"))
spkTimes=stimulated['spk_session']
stim_times=np.array(stimulated['stim_times'])
stim_durations=np.array(stimulated['stim_durations'])
stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the pythonic convention that the first channel is labeled as zero. 
afferent=stim_chs[0]
ch_inds=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
stimCh=afferent
pulseStarts=stim_times[ch_inds]
pulseDurations=stim_durations[ch_inds]

binSize=5
preCushion=10
postCushion=4
maxLapse=500

nChannels=len(spkTimes)
assert(stimCh>=1 and stimCh<=nChannels) #stimCh is the ID of the perturbed channel under the assumption that channels are numbered from one, not zero
assert(len(pulseStarts)==len(pulseDurations))
nEvents=len(pulseStarts)
   
binEdges=np.arange(0,maxLapse,binSize)
lapses=np.cumsum(np.diff(binEdges)) #command valid also when binEdges start from an offset
nLapses=len(binEdges)-1 #=len(lapses)

# pre_allocations 

preCounts=np.ma.array(np.zeros((nLapses,nEvents,nChannels,)),mask=False)
preCounts.mask[:,:,stimCh]=True

postCounts=np.ma.array(np.zeros((nLapses,nEvents,nChannels,)),mask=False)
postCounts.mask[:,:,stimCh]=True

ks=np.ma.array(np.zeros((nLapses,nEvents,nChannels,)),mask=False)
ks.mask[:,:,stimCh]=True

preInterval=maxLapse
preCount=np.ma.array(np.zeros((nEvents,nChannels,)),mask=False)
preCount.mask[:,stimCh]=True

# Also to initialize: meanPostRates, meanPreRates, postCounts, postIncrements

for event, channel in it.product(range(nEvents),(x for x in range(nChannels) if x != stimCh)):
     
     
     firstIndex=np.where(spkTimes[channel]>=pulseStarts[event]-preCushion-maxLapse)[0][0]
     lastIndex=np.where(spkTimes[channel]<pulseStarts[event]+pulseDurations[event]+postCushion+maxLapse)[0][-1]
     times=spkTimes[channel][firstIndex:lastIndex+1]-pulseStarts[event]

     postCounts[:,event,channel]=np.histogram(times,pulseDurations[event]+postCushion+binEdges)[0]
     preCounts[:,event,channel]=np.histogram(times,-preCushion-np.flip(binEdges))[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
     preCount[event,channel]=np.histogram(times,[-preInterval-preCushion,-preCushion])[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
     
#          for lapseInd,lapse in enumerate(lapses):
#               
#               postISIs=np.diff(times[(pulseDurations[event]+postCushion <times)&(times< pulseDurations[event]+postCushion+lapses[lapseInd])])
#               preISIs=np.diff(times[(-preCushion-lapses[lapseInd] <times)&(times<-preCushion)])
#
#               if min(len(postISIs),len(preISIs))>0:
#                    ks[lapseInd,event,channel]=stats.ks_2samp(preISIs,postISIs)[0] # the [1] output of ks_2samp is the p-value
#               else:
#                    ks.mask[lapseInd,event,channel]=True

incrementsByBin=postCounts/binSize-preCount/preInterval #summed through broadcasting 
mean_incrByBin=np.mean(incrementsByBin,1) #statistic over events
median_incrByBin=np.median(incrementsByBin,1) #statistic over events
std_incrByBin=np.std(incrementsByBin,1)#statistic over events

incrementsByLapse=np.cumsum(postCounts,0) #1
incrementsByLapse=np.transpose(incrementsByLapse, (1, 2, 0)) #2
incrementsByLapse=incrementsByLapse/lapses #3
incrementsByLapse=np.transpose(incrementsByLapse, (2,0,1)) #4     
incrementsByLapse=incrementsByLapse-preCount/preInterval

mean_incrByLapse=np.mean(incrementsByLapse,1)#statistic over events
median_incrByLapse=np.median(incrementsByLapse,1)
std_incrByLapse=np.std(incrementsByLapse,1)#statistic over events

wilcoxW=np.zeros((nLapses,nChannels))
wilcoxP=np.zeros((nLapses,nChannels))
     
for lapseInd, channel in it.product(range(nLapses),(x for x in range(nChannels) if x != stimCh)):
     
     wilcoxW[lapseInd,channel],wilcoxP[lapseInd,channel]=stats.wilcoxon(incrementsByLapse[lapseInd,:,channel])
     
# meanPreRate=np.mean(preCount,0)/preInterval
# meanPostRates_byBin=np.mean(postCounts,0)/binSize
# meanPostRates_byBulk=np.cumsum(np.mean(postCounts,0),1)/binSize
#  meanIncrements = masked array of length nChannels with mean rate increment over trials  (channel = afferent is masked)

mean_ks=np.mean(ks,1)#statistic over events
median_ks=np.median(ks,1)
std_ks=np.std(ks,1)#statistic over events
     
 