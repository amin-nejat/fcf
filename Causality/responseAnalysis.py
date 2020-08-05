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
import pickle
from DataTools import utilities as util
from DelayEmbedding import DelayEmbedding as DE 


# %%

def analyzeInterventions(spkTimes,stimCh,pulseStarts,pulseDurations,
          binSize=10, 
          preCushion=10, 
          postCushion=4,
          maxLapse=200,
          pval_threshold=1):
     
     
     ## if you want to convert spkTimes to rates
     # bin Size = 10 ## or larger 
     #stim_times=np.round((np.array(spkTimes)-offset)/binSize)
     #stim_durations=np.round(np.array(stim_durations)/binSize)
     # rates_stim,offset=spk2rates(spk_stim,binSize=binSize) #output is a numpy array

    nChannels=len(spkTimes)
    assert(stimCh>=0 and stimCh<=nChannels-1) #stimCh is the ID of the perturbed channel under the assumption that channels are numbered from one, not zero
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

    aggr_preISIs =[[] for i in range(nLapses)]
    for i in range(nLapses):
        aggr_preISIs[i]=[[] for i in range(nChannels)]
          
    aggr_postISIs =[[] for i in range(nLapses)]
    for i in range(nLapses):
          aggr_postISIs[i]=[[] for i in range(nChannels)]

    aggregatedKs=np.ma.array(np.zeros((nLapses,nChannels,)),mask=False)
    aggregatedKs.mask[:,stimCh]=True

     
    preInterval=maxLapse
    preCount=np.ma.array(np.zeros((nEvents,nChannels,)),mask=False)
    preCount.mask[:,stimCh]=True
    
    for event, channel in it.product(range(nEvents),(x for x in range(nChannels) if x != stimCh)):
        if len(spkTimes[channel]) == 0:
            print('No Spikes at Channel ' + str(channel))
            continue
        firstIndex=np.where(spkTimes[channel]>=pulseStarts[event]-preCushion-maxLapse)[0]
        lastIndex=np.where(spkTimes[channel]<pulseStarts[event]+pulseDurations[event]+postCushion+maxLapse)[0]
        
        if len(firstIndex) == 0 or len(lastIndex) == 0:
            print('There is no spikes in the duration considered for channel ' + str(channel))
            continue;
        else:
            firstIndex = firstIndex[0]
            lastIndex = lastIndex[-1]
        
        times=spkTimes[channel][firstIndex:lastIndex+1]-pulseStarts[event]

        postCounts[:,event,channel]=np.histogram(times,pulseDurations[event]+postCushion+binEdges)[0]
        preCounts[:,event,channel]=np.histogram(times,-preCushion-np.flip(binEdges))[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
        preCount[event,channel]=np.histogram(times,[-preInterval-preCushion,-preCushion])[0]  #pre-stimulus spike rate computed over the maximal duration used for the post-stimulus rate
          
        for lapseInd,lapse in enumerate(lapses):
               
            postISIs=np.diff(times[(pulseDurations[event]+postCushion <times)&(times< pulseDurations[event]+postCushion+lapses[lapseInd])])
            preISIs=np.diff(times[(-preCushion-lapses[lapseInd] <times)&(times<-preCushion)])
            
            if min(len(postISIs),len(preISIs))>0:
                ks[lapseInd,event,channel]=stats.mstats.ks_2samp(preISIs,postISIs)[0] # the [1] output of ks_2samp is the p-value
            else:
                ks.mask[lapseInd,event,channel]=True
               
            aggr_preISIs[lapseInd][channel]+=preISIs.tolist()
            aggr_postISIs[lapseInd][channel]+=postISIs.tolist()
               
    incrementsByBin=postCounts/binSize-preCount/preInterval #summed through broadcasting 

    incrementsByLapse=np.cumsum(postCounts,0) #1
    incrementsByLapse=np.transpose(incrementsByLapse, (1, 2, 0)) #2
    incrementsByLapse=incrementsByLapse/lapses #3
    incrementsByLapse=np.transpose(incrementsByLapse, (2,0,1)) #4     
    incrementsByLapse=incrementsByLapse-preCount/preInterval

    wilcoxW=np.ma.array(np.zeros((nLapses,nChannels,)),mask=False)
    wilcoxW.mask[:,stimCh]=True
    wilcoxP=np.ma.array(np.zeros((nLapses,nChannels,)),mask=False)
    wilcoxP.mask[:,stimCh]=True
     
    for lapseInd, channel in it.product(range(nLapses),(x for x in range(nChannels) if x != stimCh)):
        try:
            wilcoxW[lapseInd,channel],wilcoxP[lapseInd,channel]=stats.wilcoxon(incrementsByLapse.data[lapseInd,:,channel])
        except:
            
            continue;
        if min(len(aggr_preISIs[lapseInd][channel]),len(aggr_postISIs[lapseInd][channel]))>0:
            aggregatedKs[lapseInd,channel],ps=\
            stats.mstats.ks_2samp(np.array(aggr_preISIs[lapseInd][channel]),np.array(aggr_postISIs[lapseInd][channel]))
            if ps > pval_threshold:
                aggregatedKs.mask[lapseInd,channel] = True
        else:
            aggregatedKs.mask[lapseInd,channel]=True                                             
               
    responses={"wilcoxW":wilcoxW,"wilcoxP":wilcoxP,"aggregatedKs":aggregatedKs}
         
    responses["incrByBin_mean"]=np.mean(incrementsByBin,1) #statistic over events
    responses["incrByBin_median"]=np.median(incrementsByBin,1) #statistic over events
    responses["incrByBin_std"]=np.std(incrementsByBin,1)#statistic over events
    
    responses["incrByLapse_mean"]=np.mean(incrementsByLapse,1)#statistic over events
    responses["incrByLapse_median"]=np.median(incrementsByLapse,1)
    responses["incrByLapse_std"]=np.std(incrementsByLapse,1)#statistic over events
     
    responses["absIncrByLapse_mean"]=np.mean(np.abs(incrementsByLapse),1)#statistic over events
    responses["absIncrByLapse_median"]=np.median(np.abs(incrementsByLapse),1)
    responses["absIncrByLapse_std"]=np.std(np.abs(incrementsByLapse),1)#statistic over events

    responses["ks_mean"]=np.mean(ks,1)#statistic over events
    responses["ks_median"]=np.median(ks,1)
    responses["ks_std"]=np.std(ks,1)#statistic over events

     ## extra options (to initialize before the loop):
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

def causalityVsResponse(resp_measures,
                        causalPower,
                        lapses,
                        savingFilename="outputFigure",
                        corrMethod="pearson",
                        return_output=0):
     
     if corrMethod=="pearson":
          correlate=lambda x,y: stats.pearsonr(x,y)
     elif corrMethod=="spearman":
          correlate=lambda x,y: stats.spearmanr(x,y)
          
     (nLapses,nChannels)=resp_measures.shape
     assert(len(lapses)==nLapses)
     assert(nChannels==len(causalPower))     

     corrcoefs=np.zeros(nLapses)
     pValues=np.zeros(nLapses)
     meanResp=np.zeros(nLapses)
     stdResp=np.zeros(nLapses)

     corrcoefs_plus=np.zeros(nLapses)
     pValues_plus=np.zeros(nLapses)
     meanResp_plus=np.zeros(nLapses)

     corrcoefs_minus=np.zeros(nLapses)
     pValues_minus=np.zeros(nLapses)
     meanResp_minus=np.zeros(nLapses)

     
     for lapseInd in range(nLapses):

          indsAll=~resp_measures.mask[lapseInd,:]
          
          indsPlus=np.logical_and(indsAll,resp_measures.data[lapseInd,:]>0)
          
          indsMinus=np.logical_and(indsAll, resp_measures.data[lapseInd,:]<0)

          meanResp[lapseInd]=np.mean(resp_measures.data[lapseInd,indsAll])
          stdResp[lapseInd]=np.std(resp_measures.data[lapseInd,indsAll])
          meanResp_plus[lapseInd]=np.mean(resp_measures.data[lapseInd,indsPlus])
          meanResp_minus[lapseInd]=np.mean(resp_measures.data[lapseInd,indsMinus])

          corrcoefs[lapseInd],pValues[lapseInd]=correlate(
                    resp_measures.data[lapseInd,:][indsAll], causalPower[indsAll]
                    )
          
          corrcoefs_plus[lapseInd],pValues_plus[lapseInd]=correlate(
                    resp_measures.data[lapseInd,indsPlus], causalPower[indsPlus]
                    )
                    
          corrcoefs_minus[lapseInd],pValues_minus[lapseInd]=correlate(
                    resp_measures.data[lapseInd,indsMinus], causalPower[indsMinus]
                    )
          
     # make figure     
     
     fig = plt.figure()

     cAll='tab:blue'
     cPlus='tab:green'
     cMinus='tab:orange'

     ax1 = fig.add_subplot(3, 1, 1)
     ax1.errorbar(lapses,meanResp,yerr=stdResp,c=cAll)
     ax1.scatter(lapses,meanResp_plus,c=cPlus)
     ax1.scatter(lapses,meanResp_minus,c=cMinus)
     xmin,xmax= ax1.get_xlim() 
     ax1.hlines(0, xmin, xmax,colors='k')
     ax1.set_xlabel("post-stimulus lapse (ms)")
     ax1.set_ylabel("mean_resp")

     ax2 = fig.add_subplot(3, 1, 2)
     ax2.scatter(lapses,corrcoefs,c=cAll)
     ax2.scatter(lapses,corrcoefs_plus,c=cPlus)
     ax2.scatter(lapses,corrcoefs_minus,c=cMinus)
     xmin,xmax= ax2.get_xlim() 
     ax2.hlines(0, xmin, xmax,colors='k')
     ax2.set_xlabel("post-stimulus lapse (ms)")
     ax2.set_ylabel("corr_coef")

     ax3 = fig.add_subplot(3, 1, 3)
     ax3.scatter(lapses,np.log(pValues),c=cAll)
     ax3.scatter(lapses,np.log(pValues_plus),c=cPlus)
     ax3.scatter(lapses,np.log(pValues_minus),c=cMinus)
     xmin,xmax= ax3.get_xlim() 
     ax3.hlines(0, xmin, xmax,colors='k')
     pThresh=5*10**(-2)
     ax3.hlines(np.log(pThresh), xmin, xmax,colors='r')
     ax3.set_xlabel("post-stimulus lapse (ms)")
     ax3.set_ylabel("log(p_value)")

     # store plots
     plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=.1, hspace=.7)
     plt.savefig(savingFilename+"_corrMethod="+corrMethod+".jpg", bbox_inches='tight')
     plt.close('all') #keeping figures open after saving consumes memory 
     
     if return_output==1:
          return(corrcoefs,pValues)


def computeAndCompare(spkTimes_resting,
                      spkTimes_stim,
                      afferent,
                      pulseStarts,
                      pulseDurations,
                      lapseCeiling=1000,
                      analysisIdStr="computeAndCompare",
                      outputDirectory="../"):       

     """
          spkTimes_resting : list of 1d numpy arrays containing spk times for each neuron
          spkTimes_stim : same structure as spkTimes_resting
          afferent: which channel is being stimulated  (or what is the only one observed channel that is among the stimulated ones)
          pulseStarts: list of times when stimulation begins
          pulseDurations: list containg the duration of each stimulation
          outputDirectory:  where to save the figures
          analysisIdStr: string to include in the name of saved figures
          lapseCeiling: max reecommended length of post-stimulus deuration over which to analyze response
     """  

     assert(len(spkTimes_resting)==len(spkTimes_stim))
     nChannels=len(spkTimes_resting)
     
     log={"rateMaking_BinSize":50,"test_ratio":.02,"delayStep":1,"dim":5,"smoothing":False,"n_neighbors": 30,\
     "respDetectionTimeStep":7,"preCushion":10, "postCushion":4,"maxResponseLapse":500} #"corrMethod":"spearman" -- no, will try both

     responseOutput=analyzeInterventions(
                              spkTimes_stim,
                              afferent,
                              pulseStarts,
                              pulseDurations,
                              lapseCeiling=lapseCeiling,
                              binSize=log["respDetectionTimeStep"],
                              preCushion=log["preCushion"],
                              postCushion=log["postCushion"],
                              maxLapse=log["maxResponseLapse"]
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
     # connectivity_matrix, pValues_matrix = DE.connectivity(...) ## improved version of DE.connectivity will also return p-values that can then be used here.
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
                    mask=mask)

     ## One might want to try to rectify the causal powers: 
     #def relu(X):
     #return np.maximum(0,X)
     #causalPowers.append(relu(connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:]))

     causalPowers=connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:]

     resp_measure_names= list(responseOutput[0].keys())
     for resp_measure_name in resp_measure_names:
          for corrMethod in ["pearson","spearman"]:
                causalityVsResponse(
                               responseOutput[0][resp_measure_name],
                               causalPowers,
                               responseOutput[1]['lapses'],
                               savingFilename=outputDirectory+analysisIdStr+"_respMeasure="+resp_measure_name,
                               corrMethod=corrMethod
                               )

     filename=outputDirectory+"figuresLog_"+analysisIdStr+".p"
     pickle.dump(log, open(filename, "wb" )) # log = pickle.load(open(filename, "rb"))
     
     return()