# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 18:24:16 2020

@author: ff215
"""

import numpy as np
from DataTools import utilities as util
import pickle
# import itertools as it
from Causality.responseAnalysis import analyzeResponse
from DelayEmbedding import DelayEmbedding as DE 
     
def causalityVsResponse(ccmVsStim_dict):

     stim_ch=ccmVsStim_dict["stim_ch"] # which stimulated channel
     binEdges=ccmVsStim_dict["binEdges"] # edges (in ms) for the response grouping bins, to which a no-response group should be added.  
     meanValues=ccmVsStim_dict["meanCausalities"] # list of len(binEdges)) mean values of causation from stimulated channel to each of the ressponse groups
     deltas=ccmVsStim_dict["errorbars"] # error bars for each respsons egroup. 

#     causalPowersUnique=[]
#     pvalsUnique=[] 
#     
#     #this computes the correlation of causality with response, and plots.
#
#     # TO BE COMPLETED. Groups efferents by the strength of their response, and averages CCM over the groups. 
#
#          pvals=[pvals;pval(repelem(downstreamChs',nEvents),stimulatedCh)]
#          causalPowers=[causalPowers;F(repelem(downstreamChs',nEvents),stimulatedCh)]
#        
#          causalPowersUnique = [causalPowersUnique;squeeze(F(downstreamChs',stimulatedCh))]
#          pvalsUnique = [pvalsUnique;squeeze(pval(downstreamChs',stimulatedCh))]
#         - meanCausality= (lsit of length len(binEdges)) mean causation to each group.
#         - deltas= error bars for each group. 

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
          rates_resting=util.spk2rates(spk_resting,binSize=binSize)[0] #output is a numpy array
          
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
          restingMasked.mask[afferent-1,:] = True



          causalPowers, p_values = DE.recon_accuracy(cue=restingMasked,target=resting[afferent-1,:])

          # np.array size )nChannels-1) 

          #powers_to_stimCh=rconstructionAccuracy(cue=resting[stimCh-1,:],target=restingMasked)
          ccmVsResponse[analysis_counter]["causalPowers"]=powers_from_stimCh #adding as new entry into the dictionary
               
          analysis_counter+=1
               
     nAnalyses=len(ccmVsStim_dicts)
     for plotInd in range(nAnalyses):
          causalityVsResponse(ccmVsStim_dicts[plotInd])
     
