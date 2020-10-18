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
dataKeys=dataKeys[5:6] # only act on the first datasets -- for testing the code
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
     corrMatrix=util.correlateRates(restingRates) # in fact there's no need to compute the full matrix, sinces only a few rows will be used    

     for afferentInd,afferent in enumerate(set(stim_chs)):
          print("Stimulus # "+str(afferentInd+1))
          afferent_events=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
     
          for durationInd, pulseDuration in enumerate(set(stim_durations_rounded[afferent_events])):
               events=afferent_events[np.where(stim_durations_rounded[afferent_events] == pulseDuration)[0]]
               analysisID=dk[1]+"_stimChan="+str(afferent+1)+"_pulseLength="+str(int(pulseDuration))+"ms"

               print("Current analysis: "+analysisID+" .......")

               lapses,rhoCorr,pCorr,rhoCorr_plus,pCorr_plus,rhoCorr_minus,pCorr_minus,t_tTest,p_tTest=\
               computeAndCompare(
                         spk_resting,
                         spk_stim,
                         afferent,
                         stim_times[events],
                         stim_durations[events],
                         geometricMap=arrayMap[specimenID],
                         analysisIdStr=analysisID,
                         outputDirectory=outDirectory,
                         interchCorrs=corrMatrix[afferent,:]
                         )

               out["analysisIDs"].append(analysisID)
               out["rhoCorr"].append(rhoCorr)
               out["pCorr"].append(pCorr)
               out["rhoCorr_plus"].append(rhoCorr_plus)
               out["pCorr_plus"].append(pCorr_plus)
               out["rhoCorr_minus"].append(rhoCorr_minus)
               out["pCorr_minus"].append(pCorr_minus)
               out["t_tTest"].append(t_tTest)
               out["p_tTest"].append(p_tTest)
        
          out["lapses"]=lapses

     pickle.dump(out, open(outDirectory+"SUMMARY_OUTPUT", "wb" )) # log = pickle.load(open(filename, "rb"))
     #out= pickle.load(open('../../FIGS/SUMMARY_OUTPUT', 'rb'))

########### Figure showing selected outouts (in this case, the four p-values)

outDirectory="../../FIGS/"
measures=list(out["rhoCorr"][0])
nAnalyses=len(out["analysisIDs"])
methodsToPlot=["pCorr","pCorr_plus","pCorr_minus"]
     
for measure in measures:
     fig = plt.figure()
     fig, axs = plt.subplots(len(methodsToPlot),1)
     for plotInd,comparisonMethod in enumerate(methodsToPlot):
          for i in range(nAnalyses):
               axs[plotInd].plot(out["lapses"],np.log10(out[comparisonMethod][i][measure]), label=out["analysisIDs"][i])
               axs[plotInd].set_xlabel("post-stimulus lapse (ms)")
               axs[plotInd].set_ylabel("log10 "+comparisonMethod)
               axs[plotInd].legend()
               axs[plotInd].set_xlim(left=out["lapses"][0],right=out["lapses"][0]+2*(out["lapses"][-1]-out["lapses"][0]))         

     plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=.1, hspace=.7)
     plt.savefig(outDirectory+"summary_pVals_"+measure+".jpg", bbox_inches='tight')
     plt.close('all') #keeping figures open after saving consumes memory 
     
 