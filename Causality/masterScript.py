# -*- coding: utf-8 -*-

"""
Created on Wed Aug  5 11:15:19 2020
"""

import pickle
import numpy as np
import itertools as it
import pickle
from DataTools import utilities as util
from DelayEmbedding import DelayEmbedding as DE 
from responseAnalysis import computeAndCompare

keysLocation='../../data/dataKeys'
dataFolder='../../data/' #or any existing folder where you want to store the output
outputDirectory="../../FIGS/"
 
dataKeys = pickle.load(open(keysLocation, "rb"))
# dataKeys=dataKeys[:3] # only act on the first datasets -- for testing the code
usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

pulseLength_unit=60 #units of 60 ms

for dkInd,dk in enumerate(usableDataKeys):

     print("############### DATASET "+str(dkInd+1)+" OF "+str(len(usableDataKeys))+" ####################")

     resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
     resting=pickle.load(open(resting_filename, "rb"))
     nChannels=len(resting['spk_session'])
     spk_resting=resting['spk_session']
     stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
     stimulated=pickle.load(open(stim_filename, "rb"))
     spk_stim=stimulated['spk_session']
     stim_times=np.array(stimulated['stim_times'])
     minInterpulseTime=np.min(np.diff(stim_times)) #min interpulse time used as ceiling for studied response lapses
     stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the pythonic convention that the first channel is labeled as zero
     stim_durations=np.array(stimulated['stim_durations'])
     stim_durations_rounded=np.round(stim_durations/pulseLength_unit)*pulseLength_unit

     for afferentInd,afferent in enumerate(set(stim_chs)):
          print("########## Analyzing response to stimulus #"+str(afferentInd+1)+" ############")
          afferent_events=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
          for durationInd, pulseDuration in enumerate(set(stim_durations_rounded[afferent_events])):
               
               events=afferent_events[np.where(stim_durations_rounded[afferent_events] == pulseDuration)[0]]
          
               analysisID=dk[1]+"_stimChan="+str(afferent+1)+"_pulseLength="+str(int(pulseDuration))+"ms_"
               print("Now analyzing: "+analysisID+" .......")
               computeAndCompare(
                         spk_resting,
                         spk_stim,
                         afferent,
                         stim_times[events],
                         stim_durations[events],
                         analysisIdStr=analysisID,
                         lapseCeiling=minInterpulseTime,
                         outputDirectory=outputDirectory)       
