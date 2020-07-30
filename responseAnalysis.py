# -*- coding: utf-8 -*-

"""
Created on Wed Jul 29 13:29:24 2020
"""

import numpy as np
from utilities import spk2rates
import pickle
import itertools as it

def analyzeResponse(spkTimes,stimCh,pulseStarts,pulseDurations):

     """

     Explanation of the input :
          
     spkTimes: a list containg the nChannels spike trains
     stimCh: which channel was stimulated (one number)
     stimTimes: stimes   t < S when channel stimCh was stimulated
     stimDurations: the function causality_vs_response that returns all you need to make the barplots.  
     
     """
     
     ## if you want to convert spkTimes to rates
     # bin Size = ... 
     #stim_times=np.round((np.array(spkTimes)-offset)/binSize)
     #stim_durations=np.round(np.array(stim_durations)/binSize)
     # rates_stim,offset=spk2rates(spk_stim,binSize=binSize) #output is a numpy array

     nChannels=stimulated.shape[0]
     assert(stimCh>=1 and stimCh<=nChannels) #stimCh is the ID of the perturbed channel under the assumption that channels are numbered from one, not zero
     assert(len(pulseStarts)==len(pulseDurations))
     nEvents=len(pulseStarts)
     assert(stimTimes[-1]+stimDurations[-1]<stimulated.shape[1])
        
     durationStep=5
     maxPostDuration=750
     preCushion=10
     postCushion=4
     preInterval=100

     durationsArray=np.arange(durationStep,maxPostDuration,durationStep)
     nDurations=len(durationsArray)
 
     eventEfferent=[]
     pvals=[]
     causalPowers=[]
     preRates=[]
     inRates=[]
     meanPostRates=[]
     meanPreRates=[]
     meanInRates=[]
     causalPowersUnique=[]
     pvalsUnique=[]
     
     postRates=np.zeros((1,nDurations))
     postIncrements=np.zeros((1,Durations))
     
     preRate=nan(nEvents,nChannels);
     inRate=nan(nEvents,nChannels);
     postCounts=nan(nEvents,nChannels,nDurations);
                     
     for event, channel in it.product(range(nEvents),(x for x in range(nChannels) if x != stimCh-1)):
          
          firstIndex=find(spkTimes[channel]>=pulseStarts(eventInd)-preCushion-preInterval,1,'first')
          lastIndex=find(spkTimes[channel]<pulseStarts(eventInd)+pulseDurations(eventInd)+postCushion+maxPostDuration,1,'last')
          times=spkTimes[channel](firstIndex:lastIndex)-pulseStarts(eventInd)
                          
          preRate(eventInd,channel)=histcounts(times,[-preInterval-preCushion,-preCushion])/preInterval; #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
          inRate(eventInd,channel)=histcounts(times,[0,pulseDurations(eventInd)])/pulseDurations(eventInd)
          postCounts(eventInd,channel,:)=cumsum(histcounts(times,pulseDurations(eventInd)+postCushion:durationStep:pulseDurations(eventInd)+postCushion+maxPostDuration))
                          
          downstreamChs=setdiff(1:nChannels,stimulatedCh); #= [1:stimulatedCh-1,stimulatedCh+1:nChannels]
             
          preRatesPerStim=preRate(:,downstreamChs)
          preRates=[preRates;preRatesPerStim(:)]
          meanPreRates = [meanPreRates;squeeze(nanmean(preRatesPerStim))']
                               
          inRatesPerStim=inRate(:,downstreamChs)
          inRates=[inRates; inRatesPerStim(:)]
          meanInRates = [meanInRates;squeeze(nanmean(inRatesPerStim))']
                                         
          meanPostRates=[meanPostRates;squeeze(nanmean(postCounts(:,downstreamChs,:)))./durationsArray]
                                         
          postRateMatrix = bsxfun(@rdivide, postCounts, permute(durationsArray, [1 3 2]))
                                         
          for durationInd=1:nDurations
          postRate=squeeze(postRateMatrix(:,downstreamChs,durationInd))
          postRates{durationInd}=[postRates{durationInd};postRate(:)]
          postIncrements{durationInd}=[postIncrements{durationInd};postRate(:)-preRatesPerStim(:)]
                                         
          eventAfferent=[eventAfferent;stimulatedCh*ones(nEvents*(nChannels-1),1)]
          eventEfferent=[eventEfferent;repelem(downstreamChs',nEvents)]
                                                        
          pvals=[pvals;pval(repelem(downstreamChs',nEvents),stimulatedCh)]
          causalPowers=[causalPowers;F(repelem(downstreamChs',nEvents),stimulatedCh)]
        
          causalPowersUnique = [causalPowersUnique;squeeze(F(downstreamChs',stimulatedCh))]
          pvalsUnique = [pvalsUnique;squeeze(pval(downstreamChs',stimulatedCh))]
    
     
     """
    Outputs is a dictionary with the keys 
     
     - output["binEdges"]= bin-edges for the response groups, to which a no-response group should be added.  
     - meanCausality= (lsit of length len(binEdges)) mean causation to each group.
     - deltas= error bars for each group. 
     
     The outputs:
     
          mean causation from stimulated channel to everybody else within each response group. 
          Response group are defined as group of neurons that show response to stimulation within a certain intensity 
          (e.g. with a rate increment between delta_min and delta_max )

     The output plot is a plot of mean causation per group (y-values of the bar plot) and corresponding errorbars (delta-values ) 

     The bin edges for the grouping by response should be in input to the code or, if not, they can be determined autonomously
     (from the observed distribution of response)

     A "not post" response group should include efferents that show no response at all to the stimulation.

      outputDict={"stim_ch":stim_chs[stimChInd], # which stimulated channel
                    "binEdges":binEdges, #edges (in ms) for the response grouping bins, to which a no-response group should be added.  
                    "meanCausalities":meanValues, # list of len(binEdges)) mean values of causation from stimulated channel to each of the response groups
                    "errorbars":deltas #error bars for each response group. 
                    }

     
     """

     return(outputDict)
     
def causalityBarPlot(ccmVsStim_dict):

     stim_ch=ccmVsStim_dict["stim_ch"] # which stimulated channel
     binEdges=ccmVsStim_dict["binEdges"] # edges (in ms) for the response grouping bins, to which a no-response group should be added.  
     meanValues=ccmVsStim_dict["meanCausalities"] # list of len(binEdges)) mean values of causation from stimulated channel to each of the ressponse groups
     deltas=ccmVsStim_dict["errorbars"] # error bars for each respsons egroup. 

     # TO BE COMPLETED. Groups efferents by the strength of their response, and averages CCM over the groups. 

     return()
     
if __name__=="__main__":

     dataFolder='../data/' #or any existing folder where you want to store the output
     keysLocation='../data/dataKeys'              
     dataKeys = pickle.load(open(keysLocation, "rb"))
     binSize=50
     
     usableDataKeys= (dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0)
     ccmVsResponse=[] # this will contain all of the output dictionaries from the analysis, one from each stimulation subset
     analysis_counter=0 #this will be the growing index of ccmVsResponse
      
     for dk in usableDataKeys:

          resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
          resting=pickle.load(open(stim_filename, "rb"))
          spk_resting=resting['spk_session']
          rates_resting=spk2rates(spk_resting,binSize=binSize)[0] #output is a numpy array
          
          stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
          stimulated=pickle.load(open(stim_filename, "rb"))
          spk_stim=stimulated['spk_session']
          stim_times=stimulated['stim_times']
          stim_durations=stimulated['stim_durations']
          stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the convention taht numbers channels from one, not zero 
          
          for afferent in set(stim_chs):

               ch_inds=[i for i,x in enumerate(stim_chs) if x==afferent]
               ccmVsResponse.append(
                              analyzeResponse(
                                   spk_stim, 
                                   afferent, 
                                   stim_times[ch_inds],
                                   stim_durations[ch_inds]
                                   ) # the output of responseAnalysis is a dictionary that we are appending here.
                         )

              ccmVsResponse[analysis_counter]["stimulated_ch"]=afferent # add this info to the dictionary
               ## Using the "resting" matrix, create two arrys containing  all CCM values from and to the stimulated channel here called "afferent" 
               restingMasked = np.ma.array(rates_resting, mask=False)
               restingMasked.mask[stimCh,:] = True
               powers_from_stimCh=reconstructionAccuracy(cue=restingMasked,target=resting[afferent-1,:])
               #powers_to_stimCh=rconstructionAccuracy(cue=resting[stimCh-1,:],target=restingMasked)
               ccmVsResponse[analysis_counter]["causalPowers"]=powers_from_stimCh #adding as new entry into the dictionary
               
               analysis_ind+=1
               
     nAnalyses=len(ccmVsStim_dicts)
     for plotInd in range(nAnalyses):
          causalityBarPlot(ccmVsStim_dicts[plotInd])
     
