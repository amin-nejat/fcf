# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 13:56:50 2020
"""

import pickle 

dataKeys=[\
          #
          #
          ## data from specimen G : 
          [0,'G20191023a.FIRA.resting_ustim.PAG'],\
          #
          ['G20191106a.FIRA.resting.PAG',0],\
          #
          ['G20191107a.FIRA.resting.PAG','G20191107b.FIRA.resting.ustim.PAG'],\
          #
          ['G20191108a.FIRA.resting.PAG','G20191108b.FIRA.resting.ustim.PAG'],\
          #
          ['G20191203a.FIRA.resting.PAG','G20191203b.FIRA.resting.ustim.PAG'],\
          #
          ['G20191204a.FIRA.resting.PAG','G20191204b.FIRA.resting.ustim.PAG'],\
          #
          ['G20191205a.FIRA.resting.PAG','G20191205b.FIRA.resting.ustim.PAG'],\
          #
          ['G20191206a.FIRA.resting.PAG','G20191206b.FIRA.resting.ustim.PAG'],\
          #
     #
          ## data from specimen N : 
          # 
          ['N20191108a.FIRA.resting.PAG','N20191108b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191109a.FIRA.resting.PAG','N20191109b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191110a.FIRA.resting.PAG','N20191110b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191207a.FIRA.resting.PAG','N20191207b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191208a.FIRA.resting.PAG','N20191208b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191209a.FIRA.resting.PAG','N20191209b.FIRA.resting.ustim.PAG'],\
          #
          ['N20191210a.FIRA.resting.PAG','N20191210b.FIRA.resting.ustim.PAG'],\
          #              
          ]              

pickle.dump(dataKeys, open('../../data/dataKeys', "wb" ) )
     