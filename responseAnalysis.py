# -*- coding: utf-8 -*-

"""
Created on Wed Jul 29 13:29:24 2020

In input it takes:
     -  a matrix NxT of resting-state activity (where N is the number of observed channels)
     - a matrix NxS of stimulated activity, where S is the duration of this recording
     - the information on the time instant t < S where a certain channel n was stimulated   

The outputs:
     mean causation from stimulated channel to everybody else within each response group. 
Rsponse group are defined as group of neurons that show response to stimulation within a certain intensity 
(e.g. with a rate increment between delta_min and delta_max )

The output plot is a plot of mean causation per group (y-values of the bar plot) and corresponding errorbars (delta-values ) 

The bin edges for the grouping by response should be in input to the code or, if not, they can be determined autonomously
(from the observed distribution of response)

A "not post" response group should include efferents that show no response at all to the stimulation.

"""


import numpy as np
from utilities import spk2rates
import pickle

""""
     
     The function causality_vs_response that returns all you need to make the barplots.  
     
     Outputs of causality response is a list of dictionaries, one for each stimulated channel.
     Each such dictionary has the keys 
         
     - stim_ch= which stimulated channel
     - binEdges= bin-edges for the response groups, to which a no-response group should be added.  
     - meanCausality= (lsit of length len(binEdges)) mean causation to each group.
     - deltas= error bars for each group. 
     
"""

def causality_vs_response(resting, stimulated, stim_ch, stim_times,stim_durations):
     # stim_chan is assumed to be a single channel number)


     """
     
     TO BE COMPLETED. OUTLINE:
          
          - create a dictionary containing  all CCM values from and to stim_chan 
          - quantify the distribution of responses to stimulation for each channel, looking e.g. at intensity and delay. 

     
     """

     ccmVsStim_dict={"stim_ch":stim_chs[stimChInd], # which stimulated channel
                    "binEdges":binEdges, #edges (in ms) for the response grouping bins, to which a no-response group should be added.  
                    "meanCausalities":meanValues, # list of len(binEdges)) mean values of causation from stimulated channel to each of the response groups
                    "errorbars":deltas #error bars for each response group. 
                    }

     
     return(ccmVsStim_dict)
     
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
     ccmVsStim_dicts=[] # this will contain all of the output dictionaries from the analysis, one from each stimulation subset

     for dk in usableDataKeys:

          resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
          stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
          resting=pickle.load(open(stim_filename, "rb"))
          spk_resting=resting['spk_session']
          stimulated=pickle.load(open(stim_filename, "rb"))
          spk_stim=stimulated['spk_session']
          stim_times=stimulated['stim_times']
          stim_durations=stimulated['stim_durations']
          stim_channels=np.array(stimulated['stim_chan'])
          
          rates_resting=spk2rates(spk_resting,binSize=binSize)[0]
          rates_stim,offset=spk2rates(spk_stim,binSize=binSize)
          stim_times=np.array(stim_times)-offset
          stim_durations=np.array(stim_durations)/binSize
          
          for stimChan in set(stim_channels):

               ch_inds=[i for i,x in enumerate(stim_channels) if x==stimChan]
               ccmVsStim_dicts.append(
                    causality_vs_response(
                              rates_resting, 
                              rates_stim, 
                              stimChan, 
                              stim_times[ch_inds],
                              stim_durations[ch_inds]
                              )
                    )
     
     
     for plotInd in range(len(ccmVsStim_dicts)):
          causalityBarPlot(ccmVsStim_dicts[plotInd])
     
