# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:24:16 2020
"""
import pickle
import numpy as np
from Causality.responseAnalysis import analyzeResponse,plotHistograms,plotAllResponses,plotOneChannelResponse
     
# %%

whichDataset=2
whichAfferentInd=0

dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'              
dataKeys = pickle.load(open(keysLocation, "rb"))

assert(dataKeys[whichDataset][1]!=0)

stim_filename=dataFolder+'spikeData'+dataKeys[whichDataset][1]+'.p'
stimulated=pickle.load(open(stim_filename, "rb"))
spk_stim=stimulated['spk_session']
stim_times=np.array(stimulated['stim_times'])
stim_durations=np.array(stimulated['stim_durations'])
stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the convention taht numbers channels from one, not zero 

afferent=stim_chs[whichAfferentInd] #the first 
ch_inds=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(np.int64)

responseAnalysisOutput=analyzeResponse(spk_stim, afferent, stim_times[ch_inds],stim_durations[ch_inds])
plotHistograms(responseAnalysisOutput)
plotAllResponses(responseAnalysisOutput)
plotOneChannelResponse(responseAnalysisOutput)
# ksScore=responseAnalysisOutput["ksScore"]

