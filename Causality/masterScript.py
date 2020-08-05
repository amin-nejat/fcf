# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:15:19 2020

@author: ff215
"""

loop over all datasets within each dataset


     list all analyses made possible bu the stim dataset (that differ by stimulation channel and by short/long pulse duration)

     create analysisID_string 
     
     loop over subdataset in subdatasets 
     
          computeAndCompare(spkTime_resting,spkTimes_stim,stimCh, pulseStarts, pulseDurations,outputFolderAddress)

          # it doesnt matter that this will compute CCM multiple times

Notice that：

          spkTimes_resting : list of 1d numpy arrays containing spk times for each neuron
          spkTimes_stim : same structure as spkTimes_resting
          stimCh : which channel is being stimulated  (or what is the only one observed channel that is among the stimulated ones)
          pulseStarts: list of times when stimulation begins
          pulseDurations: list containg the duration of each stimulation
          outputFolderFolder:  where to save the figures