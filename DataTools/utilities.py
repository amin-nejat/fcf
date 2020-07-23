# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 18:50:10 2020
"""

import numpy as np 
from scipy import signal
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import fft

               
def smoother(rates):
     
     for ch in range(len(rates)):
          rates[ch]=signal.savgol_filter(rates[ch],\
               53,# window size used for filtering
               3) # order of fitted polynomial
     return(rates)

def normalizer(rates):
     #normalizing 
     meanRates=rates.mean(axis=1)
     stdRates=rates.std(axis=1)
     for k in range(len(rates)):
          rates[k,:]=(rates[k,:]-meanRates[k])/stdRates[k]
     return(rates)


def spk2rates(spkTimes,binSize=10,binsToSmooth=15):
     
     #spkTimes should be a list containing the spike train of each channel.
     nChannels=len(spkTimes)
     tMax=0
     for ch in range(nChannels):
          #splittingGap=ceil(mean(diff(trainTimes))+std(diff(trainTimes)))
          tMax=max(tMax,spkTimes[ch][-1])

     nBins=int(np.ceil(tMax/binSize))+1
     ## binning and condensing into one matrix
     rates=np.empty((nChannels,nBins))

     for ch in range(nChannels):
          for spkIndex in range(len(spkTimes[ch])):
               rates[ch,int(np.ceil(spkTimes[ch][spkIndex]/binSize))]+=1

     rates=smoother(rates)
     rates=normalizer(rates)
          
     return(rates)
     
def trim(times,tMin,tMax):
     trimmed= [[]for i in range(len(times))]
     for ch in range(len(times)):
          trimmed[ch]=times[ch][np.where(np.logical_and(times[ch]>=tMin, times[ch]<=tMax))[0]]
     return(trimmed)

def rasterPlot(spkTimes,tMin,tMax):

     trimmedSpkTimes=trim(spkTimes,tMin,tMax)
     
     #plot 
     line_offsets = np.arange(96)
     line_lengths = .1*np.ones(96)
     plt.eventplot(trimmedSpkTimes, lineoffsets=line_offsets,linelengths=line_lengths )
     plt.eventplot(trimmedSpkTimes)
     
     #save figure 
     
     timeNow=datetime.now()
     timestr=timeNow.strftime("%y%m%d_%H%M%S")
     figuresFolder='../../FIGS/' 
     figureName= "rasterPlot_" + timestr + ".png"
     plt.savefig(figuresFolder+figureName)
     plt.show()

     return()
     
def spectrum1ch(y):
     yf = fft(y)
     N=len(y)
     T=1/N
     xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
     spec=(2.0/N) * np.abs(yf[0:N//2])
     return(xf,spec)

def fourierAllChannels(rates):
     freqs = [[]for i in range(len(rates))]
     spectrum= [[]for i in range(len(rates))]
     for ch in range(len(rates)):
          freqs[ch],spectrum[ch]=spectrum1ch(rates[ch])
     return(freqs,spectrum)
     
     
