# -*- coding: utf-8 -*-

"""
Created on Wed Oct 14 19:54:03 2020
"""
import numpy as np
import matplotlib.pyplot as plt 
from DelayEmbedding import DelayEmbedding as DE 
from plottingTools import plotOverMap
from DataTools import utilities as util
import pickle

# %%

def plotCausalMap(spkTimes_resting,
                      geometricMap=None,
                      analysisIdStr="causalPowers",
                      outputDirectory="../"):
     
     nChannels=len(spkTimes_resting)
     
     log={"rateMaking_BinSize":50,"test_ratio":.1,"delayStep":1,"dim":5,"smoothing":False,"n_neighbors": 30,\
     "respDetectionTimeStep":7,"preCushion":10, "postCushion":4,"maxResponseLapse":500} #"corrMethod":"spearman" -- no, will try both

     rates_resting=util.spk2rates(spkTimes_resting,
                                  binSize=log["rateMaking_BinSize"],
                                  smoothing=log["smoothing"]
                                  )[0] #output is a numpy array
     
     connectivity_matrix= DE.connectivity(
                         rates_resting.T,
                         test_ratio=log["test_ratio"],
                         delay=log["delayStep"],
                         dim=log["dim"],
                         n_neighbors=log["n_neighbors"],
                         method='corr',
                         mask=np.zeros((nChannels, nChannels), dtype=bool)
                         )

     causalPowers=np.sum(connectivity_matrix,axis=0)
     causalPowers=np.ma.array(causalPowers,mask=False)

     fig = plt.figure() #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
     ax1 = fig.add_subplot(1, 1, 1)
     ax1.set_title("CCM")
     pixelMat_CCM=plotOverMap(causalPowers,geometricMap,sourceNode=None,show=False,printReport=False)
     ax1.imshow(pixelMat_CCM)

     savingFilename=outputDirectory+analysisIdStr
     plt.savefig(savingFilename+"_onGeometryMap.eps", bbox_inches='tight')
     plt.close('all') #keeping figures open after saving consumes memory 
 
     return()

               
######## MAIN #############

     
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
     
     for afferentInd,afferent in enumerate(set(stim_chs)):
          print("Stimulus # "+str(afferentInd+1))
          afferent_events=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
     
          for durationInd, pulseDuration in enumerate(set(stim_durations_rounded[afferent_events])):
               events=afferent_events[np.where(stim_durations_rounded[afferent_events] == pulseDuration)[0]]
               analysisID=dk[1]+"_stimChan="+str(afferent+1)+"_pulseLength="+str(int(pulseDuration))+"ms"

               print("Current analysis: "+analysisID+" .......")

               plotCausalMap(spk_resting,
                      geometricMap=arrayMap[specimenID],
                      analysisIdStr=analysisID,
                      outputDirectory=outDirectory)
 