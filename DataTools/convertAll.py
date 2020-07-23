# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 11:06:19 2020
"""

from dataLoader import loadData

sourceFolder='../../../SALEH/unsorted/' #or any folder with the content of https://drive.google.com/drive/u/0/folders/12usjkXjnhjhiiRUQAjzM4jnbcqyio8s1
targetFolder='../../data/spikeData' #or any existing folder where you want to store the output

datasets=[\
          #
     #
          ## data from specimen G : 
          #
          'G20191023a.FIRA.resting_ustim.PAG',\
          #
          'G20191106a.FIRA.resting.PAG',\
          #
          'G20191107a.FIRA.resting.PAG',\
          'G20191107b.FIRA.resting.ustim.PAG',\
          #
          'G20191108a.FIRA.resting.PAG',\
          'G20191108b.FIRA.resting.ustim.PAG',\
          #
          'G20191203a.FIRA.resting.PAG',\
          'G20191203b.FIRA.resting.ustim.PAG',\
          #
          'G20191204a.FIRA.resting.PAG',\
          'G20191204b.FIRA.resting.ustim.PAG',\
          #
          'G20191205a.FIRA.resting.PAG',\
          'G20191205b.FIRA.resting.ustim.PAG',\
          #
          'G20191206a.FIRA.resting.PAG',\
          'G20191206b.FIRA.resting.ustim.PAG',\
          #
     #
          ## data from specimen N : 
          # 
          'N20191108a.FIRA.resting.PAG',\
          'N20191108b.FIRA.resting.ustim.PAG',\
          #
          'N20191109a.FIRA.resting.PAG',\
          'N20191109b.FIRA.resting.ustim.PAG',\
          #
          'N20191110a.FIRA.resting.PAG',\
          'N20191110b.FIRA.resting.ustim.PAG',\
          #
          'N20191207a.FIRA.resting.PAG',\
          'N20191207b.FIRA.resting.ustim.PAG',\
          #
          'N20191208a.FIRA.resting.PAG',\
          'N20191208b.FIRA.resting.ustim.PAG',\
          #
          'N20191209a.FIRA.resting.PAG',\
          'N20191209b.FIRA.resting.ustim.PAG',\
          #
          'N20191210a.FIRA.resting.PAG',\
          'N20191210b.FIRA.resting.ustim.PAG'\
          #              
          ]              
              
for index,dataset in enumerate(datasets):
     print('converting'+str(index)+'of'+str(len(datasets))+'...')
     loadData(sourceFolder,dataset,targetFolder)

