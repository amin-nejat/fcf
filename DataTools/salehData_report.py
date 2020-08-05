# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 10:47:06 2020

@author: ff215
"""

import pickle
import numpy as np
from DataTools import utilities as util
     
dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'

dataKeys = pickle.load(open(keysLocation, "rb"))

log={"rateMaking_BinSize":50,"test_ratio":.02,"delayStep":1,"dim":5,"smoothing":False,"n_neighbors": 150,\
     "respDetectionTimeStep":7,"preCushion":10, "postCushion":4,"maxResponseLapse":500,\
     "corrMethod":"spearman"}

# dataKeys=dataKeys[:3] # only act on the first datasets -- for testing the code
usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

# %%

responseOutputs=[] # this will contain all of the output dictionaries from the response analysis
causalPowers=[]
analysisIdStrings=[]

for dkInd,dk in enumerate(usableDataKeys):

     print("############### DATASET "+str(dkInd+1)+" OF "+str(len(usableDataKeys))+" ####################")
     print("resting dataset : "+ dk[0])
     print("stimulation dataset : "+ dk[1])
     resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
     resting=pickle.load(open(resting_filename, "rb"))
     nChannels=len(resting['spk_session'])
     spk_resting=resting['spk_session']
     rates_resting=util.spk2rates(spk_resting,binSize=log["rateMaking_BinSize"],smoothing=log["smoothing"])[0] #output is a numpy array
     
     stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
     stimulated=pickle.load(open(stim_filename, "rb"))
     spk_stim=stimulated['spk_session']
     stim_times=np.array(stimulated['stim_times'])
     stim_durations=np.array(stimulated['stim_durations'])
     stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the pythonic convention that the first channel is labeled as zero. 
     print("stimulate channels: "+str(set(stim_chs)))
     print("pulse durations: "+str(set(stim_durations)))
     