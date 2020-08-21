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
dataFolder='../../data/' #or any existing folder where you want to store the output
outputDirectory="../../FIGS/"
dataKeys = pickle.load(open(keysLocation, "rb"))
#dataKeys=dataKeys[:3] # only act on the first datasets -- for testing the code
usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

pulseLength_unit=60 #units of 60 ms as used by Saleh

analysisIDs=[]
tTests=[]
pVals=[]

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

               lapses,tTestDict,pValDict=computeAndCompare(
                         spk_resting,
                         spk_stim,
                         afferent,
                         stim_times[events],
                         stim_durations[events],
                         geometricMap=arrayMap[specimenID],
                         analysisIdStr=analysisID,
                         outputDirectory=outputDirectory,
                         interchCorrs=corrMatrix[afferent,:]
                         )

               tTests.append(tTestDict)
               pVals.append(pValDict)
               analysisIDs.append(analysisID)

output={"analysisIDs": analysisIDs, "tTests":tTests, "pVals":pVals} 
pickle.dump(output, open(outputDirectory+"tTestsOutput", "wb" )) # log = pickle.load(open(filename, "rb"))
# output= pickle.load(open('../../FIGS/tTestsOutput', 'rb'))

##########  For each of the response measures we now plot 

#lapses=np.cumsum(np.diff(np.arange(0,500,7))) #command valid also when binEdges start from an offset

nAnalyses=len(tTests[0])
pThresh=5*10**(-2)
          
for method in tTests[0].keys():
     fig = plt.figure()
     ax1 = fig.add_subplot(2, 1, 1)
     for i in range(nAnalyses):
          ax1.plot(lapses,tTests[i][method], label=analysisIDs[i])
          ax1.set_xlabel("post-stimulus lapse (ms)")
          ax1.set_ylabel("t-test")
          ax1.legend()          
          ax1.set_xlim(left=lapses[0],right=lapses[0]+2*(lapses[-1]-lapses[0]))         

     ax2 = fig.add_subplot(2, 1, 2)
     for i in range(nAnalyses):
          ax2.plot(lapses,np.log(pVals[i][method]),label=analysisIDs[i])
          ax2.hlines(np.log(pThresh), lapses[0],lapses[-1],colors='r',linestyles='dashdot')
          ax2.set_xlabel("post-stimulus lapse (ms)")
          ax2.set_ylabel("p-value")
          ax2.set_xlim(left=lapses[0],right=lapses[0]+2*(lapses[-1]-lapses[0]))         
          ax2.legend()          
          
     plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=.1, hspace=.7)
     plt.savefig(outputDirectory+"summary_"+method+".jpg", bbox_inches='tight')
     plt.close('all') #keeping figures open after saving consumes memory 
     
