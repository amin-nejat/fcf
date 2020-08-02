# -*- coding: utf-8 -*-

"""
Created on Sun Aug  2 18:09:16 2020
"""

# %%

import pickle
import numpy as np
from DataTools import utilities as util
from Causality.responseAnalysis import analyzeResponse,causalityVsResponse
from DelayEmbedding import DelayEmbedding as DE 
     
dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'              
dataKeys = pickle.load(open(keysLocation, "rb"))
binSize=50

usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

# %%

ccmAndResponse=[] # this will contain all of the output dictionaries from the response analysis
causalPowers=[]
stimulatedChs=[]

for dkInd,dk in enumerate(usableDataKeys):

     print("dataset "+str(dkInd)+" of "+str(len(usableDataKeys)))
     resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
     resting=pickle.load(open(resting_filename, "rb"))
     nChannels=len(resting['spk_session'])
     spk_resting=resting['spk_session']
     rates_resting=util.spk2rates(spk_resting,binSize=binSize)[0] #output is a numpy array
     
     stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
     stimulated=pickle.load(open(stim_filename, "rb"))
     spk_stim=stimulated['spk_session']
     stim_times=np.array(stimulated['stim_times'])
     stim_durations=np.array(stimulated['stim_durations'])
     stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the pythonic convention that the first channel is labeled as zero. 
     # min_interstim_t=np.min(np.diff(stim_times)) 
     
     for afferentInd,afferent in enumerate(set(stim_chs)):

          print("Analyzing response to stimulus #"+str(afferentInd)+"...")

          ch_inds=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
          ccmAndResponse.append(
                         analyzeResponse(
                              spk_stim, 
                              afferent, 
                              stim_times[ch_inds],
                              stim_durations[ch_inds]
                              ) # the output of responseAnalysis is a dictionary that we are appending here.
                    )
          
          ## Using the "resting" matrix, of shape[0]=nChannels, 
          # create two arrys containing  all reconstruction_accuracy values from and to the stimulated channel here called "afferent" 
          # When reconstructing channel j from channel i 
          # The function connectivity returns a matrix F whose matrix element F[i,j]  is the accuracy obtained 
          # We want to see whether channel afferent can be reconstructed from the efferents. 
          # So what we want to have is the column j=afferent . We will also query the row j=efferent. 
          # This is done by turning to False every entry of the mask matrix that is either in this row or column, 
          # except the diagonal entry which stays true. 
          
          mask=np.ones((nChannels, nChannels), dtype=bool)          
          mask[afferent,np.r_[0:afferent,afferent+1:nChannels]]=False
          mask[np.r_[0:afferent,afferent+1:nChannels],afferent]=False
          
          print("Estimating causality from resting state data...")
          connectivity_matrix= DE.connectivity(
                    rates_resting.T,
                    test_ratio=.02,
                    delay=10,
                    dim=3,
                    n_neighbors=100,
                    method='corr',
                    mask=mask)
          # causalPowers, pValues = DE.connectivity(...)

          causalPowers.append(np.heaviside(connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:],.5))
          stimulatedChs.append(afferent)
               
# %%

print("comparing reponse to causation")
response_measure_name="mean_incrByLapse"
nAnalyses=len(ccmAndResponse)
for plotInd in range(nAnalyses):
     causalityVsResponse(ccmAndResponse[plotInd][0],
                         ccmAndResponse[plotInd][1],
                         causalPowers[plotInd],
                         stimulatedChs[plotInd],
                         response_measure_name)
                    
     
# %%
