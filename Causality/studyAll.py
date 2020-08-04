# -*- coding: utf-8 -*-

"""
Created on Sun Aug  2 18:09:16 2020
"""

# %%

import pickle
import numpy as np
from DataTools import utilities as util
from Causality.responseAnalysis import analyzeResponse,causalityVsResponse
from Causality.utilities import relu
from DelayEmbedding import DelayEmbedding as DE 
     
dataFolder='../../data/' #or any existing folder where you want to store the output
keysLocation='../../data/dataKeys'
dataKeys = pickle.load(open(keysLocation, "rb"))
log={"rateMaking_BinSize":50,"test_ratio":.02,"delayStep":1,"dim":5,"n_neighbors": 150,\
     "responseBinSize":10,"preCushion":10, "postCushion":4,"maxResponseLapse":500}

# dataKeys=dataKeys[:3] # only act on the first datasets -- for testing the code
usableDataKeys= [dk for dk in dataKeys if dk[0]!=0 and dk[1]!=0]

# %%

responseOutputs=[] # this will contain all of the output dictionaries from the response analysis
causalPowers=[]
analysisIdStrings=[]

for dkInd,dk in enumerate(usableDataKeys):

     print("############### DATASET "+str(dkInd+1)+" OF "+str(len(usableDataKeys))+" ####################")
     resting_filename=dataFolder+'spikeData'+dk[0]+'.p'
     resting=pickle.load(open(resting_filename, "rb"))
     nChannels=len(resting['spk_session'])
     spk_resting=resting['spk_session']
     rates_resting=util.spk2rates(spk_resting,binSize=log["rateMaking_BinSize"],smoothing=0)[0] #output is a numpy array
     
     stim_filename=dataFolder+'spikeData'+dk[1]+'.p'
     stimulated=pickle.load(open(stim_filename, "rb"))
     spk_stim=stimulated['spk_session']
     stim_times=np.array(stimulated['stim_times'])
     stim_durations=np.array(stimulated['stim_durations'])
     stim_chs=np.array(stimulated['stim_chan']) #warning: this gives the id of stimulated channels under the pythonic convention that the first channel is labeled as zero. 
     # min_interstim_t=np.min(np.diff(stim_times)) 
     minInterpulseTime=np.min(np.diff(stim_times))
     
     for afferentInd,afferent in enumerate(set(stim_chs)):

          print("########## Analyzing response to stimulus #"+str(afferentInd+1)+" ############")

          ch_inds=np.array([i for i,x in enumerate(stim_chs) if x==afferent]).astype(int)
          responseOutputs.append(
                         analyzeResponse(
                              spk_stim, 
                              afferent, 
                              stim_times[ch_inds],
                              stim_durations[ch_inds],
                              lapseCeiling=minInterpulseTime,
                              binSize=log["responseBinSize"], 
                              preCushion=log["preCushion"],
                              postCushion=log["postCushion"],
                              maxLapse=log["maxResponseLapse"]
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
          # connectivity_matrix, pValues_matrix = DE.connectivity(...)
          connectivity_matrix= DE.connectivity(
                    rates_resting.T,
                    test_ratio=log["test_ratio"],
                    delay=log["delayStep"],
                    dim=log["dim"],
                    n_neighbors=log["n_neighbors"],
                    method='corr',
                    mask=mask)

          #causalPowers.append(relu(connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:]))
          causalPowers.append(connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:])
          print(dk[1]+"_stimCh="+str(afferent))
          analysisIdStrings.append(dk[1]+"_stimCh="+str(afferent+1))

# %%

print("Comparing reponse to causation and outputting scatter plots ...")

resp_measure_names= list(responseOutputs[0][0].keys())

for resp_measure_name in resp_measure_names:
     print("Chosen response measure = "+resp_measure_name)
     nAnalyses=len(responseOutputs)
      
     for analysisInd in range(nAnalyses):
          print("analysis #  "+str(analysisInd )+" of "+str(nAnalyses))                
          figuresFolder="../../FIGS/"
          causalityVsResponse(
                    responseOutputs[analysisInd][0][resp_measure_name],
                    causalPowers[analysisInd],
                    responseOutputs[analysisInd][1]['lapses'],
                    figuresFolder+analysisIdStrings[analysisInd]+"_respMeasure="+resp_measure_name+".jpg",
                    return_output=0
                    )
# in the figure folder, save also the log
          
filename=figuresFolder+"figuresLog.p"
pickle.dump(log, open(filename, "wb" )) # log = pickle.load(open(filename, "rb"))
