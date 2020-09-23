# -*- coding: utf-8 -*-

"""
Created on Wed Aug  5 11:15:19 2020
"""

import pickle
import numpy as np
from responseAnalysis import computeAndCompare
from DataTools import utilities as util
from matplotlib import pyplot as plt     

######## First we extract the t-valus and p-values for each dataset
     
arrayMap= pickle.load(open('../../data/arrayMap.p', "rb")) 

keysLocation='../../data/dataKeys'
dataFolder='../../data/' #or any existing folder where you want to store the out
outDirectory="../../FIGS/"
dataKeys = pickle.load(open(keysLocation, "rb"))
#dataKeys=dataKeys[:3] # only act on the first datasets -- for testing the code
usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

pulseLength_unit=60 #units of 60 ms as used by Saleh

out={"analysisIDs": [], "lapses":[],"rhoCorr":[],"pCorr":[],"rhoCorr_plus":[],"pCorr_plus":[],
        "rhoCorr_minus":[],"pCorr_minus":[],"t_tTest":[],"p_tTest":[]} 

for dkInd,dk in enumerate(usableDataKeys):

     print("### DATASET "+str(dkInd+1)+" OF "+str(len(usableDataKeys))+" ###")
    
     specimenID=dk[0][0]
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

     restingRates=util.spk2rates(spk_resting,binSize=10,smoothing=False)[0] #should this be the same bin we use for CCM (50ms)?
     
    