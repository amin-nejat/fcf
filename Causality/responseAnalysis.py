# -*- coding: utf-8 -*-

"""
Created on Wed Jul 29 13:29:24 2020
"""

# %%

import numpy as np
import itertools as it
import matplotlib.pyplot as plt 
import matplotlib.pylab as pyl
from scipy import stats


# %%

def analyzeResponse(spkTimes,stimCh,pulseStarts,pulseDurations,
          lapseCeiling=1000,
          binSize=10, 
          preCushion=10, 
          postCushion=4,
          maxLapse=120):

     if (postCushion+maxLapse>lapseCeiling):
          print("WARNING:  lapseCeiling ="+str(lapseCeiling)+"<"+str(maxLapse))
          #this is the maximum allowed poststimulus lapse over it is legitimate to analyze
     else:         
          print("OK:  lapse ceiling ="+str(lapseCeiling)+">"+str(maxLapse))
     
     
     ## if you want to convert spkTimes to rates
     # bin Size = 10 ## or larger 
     #stim_times=np.round((np.array(spkTimes)-offset)/binSize)
     #stim_durations=np.round(np.array(stim_durations)/binSize)
     # rates_stim,offset=spk2rates(spk_stim,binSize=binSize) #output is a numpy array

     nChannels=len(spkTimes)
     assert(stimCh>=1 and stimCh<=nChannels) #stimCh is the ID of the perturbed channel under the assumption that channels are numbered from one, not zero
     assert(len(pulseStarts)==len(pulseDurations))
     nEvents=len(pulseStarts)
        
     binEdges=np.arange(0,maxLapse,binSize)
     lapses=np.cumsum(np.diff(binEdges)) #command valid also when binEdges start from an offset
     nLapses=len(binEdges)-1 #=len(lapses)

     ## pre_allocations 

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
          
          #print("event = "+str(event)+" and channel = "+str(channel))
          
          firstIndex=np.where(spkTimes[channel]>=pulseStarts[event]-preCushion-maxLapse)[0][0]
          lastIndex=np.where(spkTimes[channel]<pulseStarts[event]+pulseDurations[event]+postCushion+maxLapse)[0][-1]
          times=spkTimes[channel][firstIndex:lastIndex+1]-pulseStarts[event]

          postCounts[:,event,channel]=np.histogram(times,pulseDurations[event]+postCushion+binEdges)[0]
          preCounts[:,event,channel]=np.histogram(times,-preCushion-np.flip(binEdges))[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
          preCount[event,channel]=np.histogram(times,[-preInterval-preCushion,-preCushion])[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
          
          for lapseInd,lapse in enumerate(lapses):
               
               postISIs=np.diff(times[(pulseDurations[event]+postCushion <times)&(times< pulseDurations[event]+postCushion+lapses[lapseInd])])
               preISIs=np.diff(times[(-preCushion-lapses[lapseInd] <times)&(times<-preCushion)])

               if min(len(postISIs),len(preISIs))>0:
                    ks[lapseInd,event,channel]=stats.ks_2samp(preISIs,postISIs)[0] # the [1] output of ks_2samp is the p-value
               else:
                    ks.mask[lapseInd,event,channel]=True

     incrementsByBin=postCounts/binSize-preCount/preInterval #summed through broadcasting 

     incrementsByLapse=np.cumsum(postCounts,0) #1
     incrementsByLapse=np.transpose(incrementsByLapse, (1, 2, 0)) #2
     incrementsByLapse=incrementsByLapse/lapses #3
     incrementsByLapse=np.transpose(incrementsByLapse, (2,0,1)) #4     
     incrementsByLapse=incrementsByLapse-preCount/preInterval

     wilcoxW=np.ma.array(np.zeros((nLapses,nChannels,)),mask=False)
     wilcoxP=np.ma.array(np.zeros((nLapses,nChannels,)),mask=False)
     
     for lapseInd, channel in it.product(range(nLapses),(x for x in range(nChannels) if x != stimCh)):
          wilcoxW[lapseInd,channel],wilcoxP[lapseInd,channel]=stats.wilcoxon(incrementsByLapse.data[lapseInd,:,channel])

     responses={"wilcoxW":wilcoxW,"wilcoxP":wilcoxP}
         
     responses["mean_incrByBin"]=np.mean(incrementsByBin,1) #statistic over events
     responses["median_incrByBin"]=np.median(incrementsByBin,1) #statistic over events
     responses["std_incrByBin"]=np.std(incrementsByBin,1)#statistic over events
     
     responses["mean_incrByLapse"]=np.mean(incrementsByLapse,1)#statistic over events
     responses["median_incrByLapse"]=np.median(incrementsByLapse,1)
     responses["std_incrByLapse"]=np.std(incrementsByLapse,1)#statistic over events
     
     responses["mean_AbsIncrByLapse"]=np.mean(np.abs(incrementsByLapse),1)#statistic over events
     responses["median_AbsIncrByLapse"]=np.median(np.abs(incrementsByLapse),1)
     responses["std_AbsIncrByLapse"]=np.std(np.abs(incrementsByLapse),1)#statistic over events

     responses["mean_ks"]=np.mean(ks,1)#statistic over events
     responses["median_ks"]=np.median(ks,1)
     responses["std_ks"]=np.std(ks,1)#statistic over events

     # responses["meanPreRate"]=np.mean(preCount,0)/preInterval
     # responses["meanPostRates_byBin"]=np.mean(postCounts,0)/binSize
     # responses["meanPostRates_byBulk"]=np.cumsum(np.mean(postCounts,0),1)/binSize
     # responses["meanIncrements"]= masked array of length nChannels with mean rate increment over trials  (channel = afferent is masked)

     log={"nChannels":nChannels,"nEvents":nEvents,"binSize":binSize,\
          "nLapses":nLapses,"preCushion":preCushion,"postCushion":postCushion,\
           "preInterval":preInterval,"maxLapse":maxLapse,"binEdges":binEdges,"lapses":lapses,\
           "incrementsByLapse":incrementsByLapse,"incrementsByBin":incrementsByBin,"ks":ks,\
           "stimulated_channel":stimCh}
     
     return(responses,log)

# %%

def plotHistograms(analyzeResponse_output):
     # input is the output of the analyzeResponse method
         
     nBinsToPlot=25
     colors = pyl.cm.brg(np.linspace(0,1,nBinsToPlot))
     incrs=analyzeResponse_output[0]["incrementsByLapse"] # incrs.shape = (nLapses,nEvents,nChannels,)
     
     nChannelsToPlot=3
     plt.title("rate increment")

     for channel in range(nChannelsToPlot):
          plt.title("effect of the stimulation of ch"+str(analyzeResponse_output[1]["stimulated_channel"])+"on ch"+str(channel+1))
          for lapseInd in range(nBinsToPlot):
               counts,edges=np.histogram(incrs[lapseInd ,:,channel])
               midpoints=(edges[1:]+edges[:-1])/2
               plt.plot(midpoints,counts, color=colors[lapseInd], label=str(analyzeResponse_output[1]["binEdges"][lapseInd+1])+" ms")
               plt.legend()
               #plt.savefig("histograms")
          plt.show()
         
     return()
     
# %%

def plotAllResponses(analyzeResponse_output):          
     # input is the output of the analyzeResponse method

     edges=analyzeResponse_output[1]["binEdges"]
     midpoints=(edges[:-1]+edges[1:])/2             

     nChannels=96
     plt.plot(midpoints,analyzeResponse_output[0]["mean_incrByLapse"][:,0:nChannels])
     plt.xlabel("time (ms)")
     plt.ylabel("rate increment")
     
     plt.plot(midpoints,analyzeResponse_output[0]["mean_incrByLapse"][:,0])
     plt.title("response of different channels to channel "+str(analyzeResponse_output[1]["stimulated_channel"]))
     x_left, x_right = plt.xlim()      
     plt.hlines(0, x_left, x_right)
     plt.show()     

     return()

# %%

def plotOneChannelResponse(analyzeResponse_output):

     # input is the output of the analyzeResponse method
     efferent=1
     plt.title("rate increment")
     incrs=analyzeResponse_output[0]["incrementsByLapse"] # incrs.shape = (nLapses,nEvents,nChannels,)
     nLapses,nEvents,nChannels = incrs.shape
     edges=analyzeResponse_output[1]["binEdges"]
     plt.title("trials of stimulating ch"+str(analyzeResponse_output[1]["stimulated_channel"])+", recording ch"+str(efferent+1))
                   
     for event in range(nEvents):
        plt.plot(edges[1:],incrs[:,event,efferent])
     plt.xlabel("post-stimulation lapse")
     plt.ylabel("rate increment")
     x_left, x_right = plt.xlim()      
     plt.hlines(0, x_left, x_right)
     plt.show()
     return()
     
# %%

def causalityVsResponse(resp_measures,causalPower,lapses,savingFilename,return_pValues=0):
     
     (nLapses,nChannels)=resp_measures.shape
     assert(len(lapses)==nLapses)
     assert(nChannels==len(causalPower))     
     corrcoefs=np.zeros(nLapses)
     pValues=np.zeros(nLapses)
     meanRespMeasure=np.zeros(nLapses)
     stdRespMeasure=np.zeros(nLapses)
     
     for lapseInd in range(nLapses):
          unmasked=~resp_measures.mask[lapseInd,:]
          meanRespMeasure[lapseInd]=np.mean(resp_measures.data[lapseInd,:][unmasked])
          stdRespMeasure[lapseInd]=np.std(resp_measures.data[lapseInd,:][unmasked])
          corrcoefs[lapseInd],pValues[lapseInd]=stats.pearsonr(resp_measures.data[lapseInd,:][unmasked], causalPower[unmasked])
          
     # make figure     
     
     fig = plt.figure()

     ax1 = fig.add_subplot(3, 1, 1)
     ax1.errorbar(lapses,meanRespMeasure,yerr=stdRespMeasure)
     ax1.scatter(lapses,meanRespMeasure)
     ax1.set_xlabel("post-stimulus lapse (ms)")
     ax1.set_ylabel("mean_resp")

     ax3 = fig.add_subplot(3, 1, 2)
     ax3.scatter(lapses,corrcoefs)
     ax3.set_xlabel("post-stimulus lapse (ms)")
     ax3.set_ylabel("corr_coef")

     ax4 = fig.add_subplot(3, 1, 3)
     ax4.scatter(lapses,np.log(pValues))
     xmin,xmax= ax1.get_xlim() 
     pThresh=5*10**(-2)
     plt.hlines(np.log(pThresh), xmin, xmax,colors='r')
     ax4.set_xlabel("post-stimulus lapse (ms)")
     ax4.set_ylabel("log(p_value)")

     # save and show plots
     fig.tight_layout()
     plt.savefig(savingFilename, bbox_inches='tight')
     fig.show()
     if return_pValues==1:
          return(pValues)
