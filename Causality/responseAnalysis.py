# -*- coding: utf-8 -*-

"""
Created on Wed Jul 29 13:29:24 2020
"""

# %%

import numpy as np
import itertools as it
import matplotlib.pyplot as plt 
from scipy import stats
import pickle
from DataTools import utilities as util
from DelayEmbedding import DelayEmbedding as DE 
from itertools import groupby
from operator import itemgetter
from .plottingTools import plotOverMap
from copy import deepcopy
from scipy.io import savemat
# %%

def interventional_connectivity(activity,stim,t=None,bin_size=10,skip_pre=10,skip_pst=4,pval_threshold=1,methods=['mean_isi','aggr_ks','mean_ks','aggr_ks_pval'],save_data=False,file=None):
    # stim is an array of tuples [(chn_1,str_1,end_1),(chn_1,str_,end_1),...]
    stim_ = deepcopy(stim)
    
    for i in range(len(stim)):
        if t is None:
            pst_isi = [np.diff(activity[j][(activity[j] <  stim_[i][2]+skip_pst+bin_size) & (activity[j] >= stim_[i][2]+skip_pst)]) for j in range(len(activity))]
            pre_isi = [np.diff(activity[j][(activity[j] >= stim_[i][1]-skip_pre-bin_size) & (activity[j] <  stim_[i][1]-skip_pre)]) for j in range(len(activity))]
        else:
            pst_isi = [activity[j][(t <  stim_[i][2]+skip_pst+bin_size) & (t >= stim_[i][2]+skip_pst)] for j in range(len(activity))]
            pre_isi = [activity[j][(t >= stim_[i][1]-skip_pre-bin_size) & (t <  stim_[i][1]-skip_pre)] for j in range(len(activity))]
        
        stim_[i] += (pre_isi,pst_isi)
    
    stim_g = [(k, [(x3,x4) for _,x1,x2,x3,x4 in g]) for k, g in groupby(sorted(stim_,key=itemgetter(0)), key=itemgetter(0))]
    
    output = {}
    count = {}
    for m in methods:
        output[m] = np.zeros((len(activity), len(activity)))*np.nan
        count[m] = np.zeros((len(activity), len(activity)))*.0
    
    for i in range(len(stim_g)): # stimulation channel
        print('Computing intervention effect for channel ' + str(i))
        for n in range(len(activity)): # post-syn channel
            aggr_pre_isi = []
            aggr_pst_isi = []
            for j in range(len(stim_g[i][1])): # stimulation event
                if 'mean_ks' in methods:
                    if len(stim_g[i][1][j][0][n]) > 0 and len(stim_g[i][1][j][1][n]) > 0:
                        ks,p = stats.mstats.ks_2samp(stim_g[i][1][j][0][n],stim_g[i][1][j][1][n])
                        if p <= pval_threshold:
                            output['mean_ks'][stim_g[i][0],n] = np.nansum((output['mean_ks'][stim_g[i][0],n],ks))
                            count['mean_ks'][stim_g[i][0],n] += 1
                            
                if 'mean_isi' in methods:
                    df_f = (stim_g[i][1][j][1][n].mean()-stim_g[i][1][j][0][n].mean())
                    output['mean_isi'][stim_g[i][0],n] = np.nansum((output['mean_isi'][stim_g[i][0],n],df_f))
                    count['mean_isi'][stim_g[i][0],n] += 1
                
                aggr_pre_isi.append(stim_g[i][1][j][0][n])
                aggr_pst_isi.append(stim_g[i][1][j][1][n])
            
            if 'aggr_ks' in methods:
                if np.array(aggr_pre_isi).size > 0 and np.array(aggr_pst_isi).size > 0:
                    ks,p = stats.mstats.ks_2samp(np.hstack(aggr_pre_isi),np.hstack(aggr_pst_isi))
                    if p <= pval_threshold:
                        output['aggr_ks'][stim_g[i][0]][n] = ks
                        count['aggr_ks'][stim_g[i][0]][n] = 1
                        if 'aggr_ks_pval' in methods:
                            output['aggr_ks_pval'][stim_g[i][0]][n] = p
                            count['aggr_ks_pval'][stim_g[i][0]][n] = 1
        for m in methods:
            output[m] /= count[m]
    
    if save_data:
        savemat(file+'.mat',{'activity':activity,'stim':stim,'t':t,'bin_size':bin_size,
                             'skip_pre':skip_pre,'skip_pst':skip_pst,'pval_threshold':pval_threshold,
                             'methods':methods,'output':output})
            
    return output


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
    #responses["ks_median"]=np.median(ks,1)
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

def compareByCausalityGroup(
                        resp_measures,
                        causalPower,
                        lapses,
                        savingFilename="outputFigure",
                        return_output=True,
                        makePlots=True):
     
     #Here the analysis, which consists in grouping depending on the sign of the causal power and comparing responses with a t-test.
     nLapses,nChannels=resp_measures.shape
     assert(len(causalPower)==nChannels)  
     assert(len(lapses)==nLapses)

     meanDownstream=np.mean(resp_measures[:,causalPower>0],1)
     meanUpstream=np.mean(resp_measures[:,causalPower<0],1)
     stdDownstream=np.std(resp_measures[:,causalPower>0],1)
     stdUpstream=np.std(resp_measures[:,causalPower<0],1)
     
     tTest=np.zeros(nLapses)
     pValue=np.zeros(nLapses)
     for lapseInd in range(nLapses):
          tTest[lapseInd],pValue[lapseInd]=stats.ttest_ind(resp_measures[lapseInd,causalPower>0],resp_measures[lapseInd,causalPower<0])
          
     meanResp=np.zeros(nLapses)
     stdResp=np.zeros(nLapses)
     meanResp_plus=np.zeros(nLapses)
     meanResp_minus=np.zeros(nLapses)

     for lapseInd,lapse in enumerate(lapses):

          indsAll=~resp_measures.mask[lapseInd,:]
          indsPlus=np.logical_and(indsAll,resp_measures.data[lapseInd,:]>0)
          indsMinus=np.logical_and(indsAll, resp_measures.data[lapseInd,:]<0)
          
          meanResp[lapseInd]=np.mean(resp_measures.data[lapseInd,indsAll])
          stdResp[lapseInd]=np.std(resp_measures.data[lapseInd,indsAll])
          meanResp_plus[lapseInd]=np.mean(resp_measures.data[lapseInd,indsPlus])
          meanResp_minus[lapseInd]=np.mean(resp_measures.data[lapseInd,indsMinus])

     if makePlots==True:
          
          fig = plt.figure()
          
          ax1 = fig.add_subplot(4, 1, 1)
    
          ax1.errorbar(lapses,meanResp,yerr=stdResp,label="all channels")
          ax1.scatter(lapses,meanResp_plus,label="positively responding chs")
          ax1.scatter(lapses,meanResp_minus,label="negatively responding chs")
          xmin,xmax= ax1.get_xlim() 
          ax1.hlines(0, xmin, xmax,colors='k')
          ax1.set_xlabel("post-stimulus lapse (ms)")
          ax1.set_ylabel("mean_resp")
          ax1.legend()          
     
          ax2 = fig.add_subplot(4, 1, 2)
          ax2.errorbar(lapses,meanDownstream,yerr=stdDownstream,label="putative upstream channels")
          ax2.errorbar(lapses,meanUpstream,yerr=stdUpstream,label="putative downstream channels")
          ax2.errorbar(lapses,meanResp,yerr=stdResp,label="all channels")
          xmin,xmax= ax2.get_xlim() 
          ax2.hlines(0, xmin, xmax,colors='k')
          ax2.set_xlabel("post-stimulus lapse (ms)")
          ax2.set_ylabel("mean_resp")
          ax2.legend()          
     
          ax3 = fig.add_subplot(4, 1, 3)
          ax3.scatter(lapses,tTest)
          xmin,xmax= ax3.get_xlim() 
          ax3.hlines(0, xmin, xmax,colors='k')
          pThresh=5*10**(-2)
          ax3.set_xlabel("post-stimulus lapse (ms)")
          ax3.set_ylabel("t-test")
          
          ax4 = fig.add_subplot(4, 1, 4)
          ax4.scatter(lapses,np.log(pValue))
          xmin,xmax= ax4.get_xlim() 
          ax4.hlines(0, xmin, xmax,colors='k')
          pThresh=5*10**(-2)
          ax4.hlines(np.log(pThresh), xmin, xmax,colors='r')
          ax4.set_xlabel("post-stimulus lapse (ms)")
          ax4.set_ylabel("log(p_value)")

          # store plots
          plt.subplots_adjust(left=0, right=2, bottom=0, top=2, wspace=.1, hspace=.7)
          plt.savefig(savingFilename+"_tTest.jpg", bbox_inches='tight')
          plt.close('all') #keeping figures open after saving consumes memory 


     if return_output==True:
          return(tTest,pValue)


def compareByLapse(resp_measures,
                        causalPower,
                        lapses,
                        savingFilename="outputFigure",
                        corrMethod="pearson",
                        return_output=False,
                        makePlots=True):

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

          corrcoefs[lapseInd],pValues[lapseInd]=correlate(resp_measures.data[lapseInd,:][indsAll], causalPower[indsAll])
          corrcoefs_plus[lapseInd],pValues_plus[lapseInd]=correlate(resp_measures.data[lapseInd,indsPlus], causalPower[indsPlus])
          corrcoefs_minus[lapseInd],pValues_minus[lapseInd]=correlate(resp_measures.data[lapseInd,indsMinus], causalPower[indsMinus])
          
     # make figure     
     if makePlots==True:
          
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
          
     if return_output==True:
          return(corrcoefs,pValues,corrcoefs_plus,pValues_plus,corrcoefs_minus,pValues_minus)
          
# %%

def compareByGeometry(
          responses,
          causalPowers,
          geometricMap,
          afferent=None,
          corrVec=None,
          savingFilename="../",
          titleString="causal map comparison"):

     
     """
           - plot the interventional response Vector and the ccm Vector for a given affernt using plotOverMap
           - save the two-panel figures into the prescribed outputDirectory

     """

     if corrVec==None:
          corrVec=np.zeros(causalPowers.shape)

     fig = plt.figure() #fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)
     ax1 = fig.add_subplot(1, 3, 1)
     ax1.set_title("CCM")
     pixelMat_CCM=plotOverMap(causalPowers,geometricMap,sourceNode=afferent,show=False,printReport=False)
     ax1.imshow(pixelMat_CCM)
     
     ax2 = fig.add_subplot(1, 3, 2)
     ax2.set_title("intervention")
     pixelMat_interv=plotOverMap(responses.data,geometricMap,sourceNode=afferent,show=False,printReport=False)
     ax2.imshow(pixelMat_interv)
     
     ax3 = fig.add_subplot(1, 3, 3)
     ax3.set_title("correlations")
     pixelMat_corr=plotOverMap(corrVec,geometricMap,sourceNode=afferent,show=False,printReport=False)
     ax3.imshow(pixelMat_corr)
     
     fig.suptitle(titleString)

    
     #save figure
     plt.savefig(savingFilename+"_onGeometryMap.jpg", bbox_inches='tight')
     plt.close('all') #keeping figures open after saving consumes memory 
     
     return()

# %%
          
def computeAndCompare(spkTimes_resting,
                      spkTimes_stim,
                      afferent,
                      pulseStarts,
                      pulseDurations,
                      geometricMap=None,
                      analysisIdStr="computeAndCompare",
                      outputDirectory="../",
                      corrMethod="pearson",
                      interchCorrs=None,   
                      makePlots=False,
                      plotOnGeometry=False):
     
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

     print("...analyzing interventions..........")
     responseOutput=analyzeInterventions(
                              spkTimes_stim,
                              afferent,
                              pulseStarts,
                              pulseDurations,
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
          
     print("...estimating causality from resting state data...")
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
                         mask=mask
                         )

     ## One might want to try to rectify the causal powers: 
     # def relu(X):
     # return np.maximum(0,X)
     # causalPowers.append(relu(connectivity_matrix[:,afferent] -connectivity_matrix[afferent,:]))

     causalPowers=connectivity_matrix[:,afferent]-connectivity_matrix[afferent,:]
     causalPowers=np.ma.array(causalPowers,mask=False)
     causalPowers.mask[afferent]=True
          
     print("Correlating ccm causal powers with interventional weights...")

     rhoCorr={}
     pCorr={}
     rhoCorr_plus={}
     pCorr_plus={}
     rhoCorr_minus={}
     pCorr_minus={}
     t_tTest={}
     p_tTest={}
     
     resp_measure_names= list(responseOutput[0].keys())
     lapses=responseOutput[1]["lapses"]

     for resp_measure_name in resp_measure_names:

          print('...'+resp_measure_name)
                
          print("....... Performing t-tests group response of CCM's putative downstream/upstream channels:")
          t_tTest[resp_measure_name],p_tTest[resp_measure_name]=compareByCausalityGroup(
                    responseOutput[0][resp_measure_name],
                    causalPowers,
                    responseOutput[1]['lapses'],
                    savingFilename=outputDirectory+analysisIdStr+"_respMeasure="+resp_measure_name,
                    return_output=True,
                    makePlots=makePlots
                    )
     

          print("....... Correlating response with ccm over a range of response lapses...")

          rhoCorr[resp_measure_name],pCorr[resp_measure_name],\
          rhoCorr_plus[resp_measure_name],pCorr_plus[resp_measure_name],\
          rhoCorr_minus[resp_measure_name],pCorr_minus[resp_measure_name]=\
                 compareByLapse(
                    responseOutput[0][resp_measure_name],
                    causalPowers,
                    responseOutput[1]['lapses'],
                    savingFilename=outputDirectory+analysisIdStr+"_respMeasure="+resp_measure_name,
                    corrMethod=corrMethod,
                    return_output=True,
                    makePlots=makePlots
                    )

          if plotOnGeometry==True:
     
               print("......  Producing figure on the geometrical map...")
     
               optimal_pValue=np.min(p_tTest[resp_measure_name])
               optimalLapseInd=np.argmin(p_tTest[resp_measure_name])
               optimalLapse=responseOutput[1]['lapses'][optimalLapseInd]
               titleString="p-value="+str(round(optimal_pValue,5))+" at "+str(optimalLapse)+"ms with "+resp_measure_name
     
               mask = np.ones(causalPowers.shape,dtype=bool)
               mask[afferent]=0
               normalizer=np.max(causalPowers[mask])
     
               compareByGeometry(
                         responseOutput[0][resp_measure_name][optimalLapseInd,:],
                         causalPowers/normalizer,
                         geometricMap,
                         afferent=afferent,
                         savingFilename=outputDirectory+analysisIdStr+"_respMeasure="+resp_measure_name+"_onMap",
                         titleString=titleString,
                         corrVec=interchCorrs)

     # saving log inside the figure folder
     filename=outputDirectory+"figuresLog_"+analysisIdStr+".p"
     pickle.dump(log, open(filename, "wb" )) # log = pickle.load(open(filename, "rb"))

     return(lapses,rhoCorr,pCorr,rhoCorr_plus,pCorr_plus,rhoCorr_minus,pCorr_minus,t_tTest,p_tTest)
