# -*- coding: utf-8 -*-

"""
Created on Tue Aug 11 09:25:36 2020
"""

import numpy as np
import pickle 

def array_maps(save=False):
    arrayMap={}
    
    ## the followings are the original array map matrices, with the matlab convention of the lowest channel index being 1
    
    arrayMap["G"]=np.array([[3,2,1,0,4,6,8,0,14,10],\
                           [65,66,33,34,7,9,11,12,16,18],\
                           [67,68,35,36,5,17,13,23,20,22],\
                           [69,70,37,38,48,15,19,25,27,24],\
                           [71,72,39,40,42,50,54,21,29,26],\
                           [73,74,41,43,44,46,52,62,31,28],\
                           [75,76,45,47,51,56,58,60,64,30],\
                           [77,78,82,49,53,55,57,59,61,32],\
                           [79,80,84,86,87,89,91,94,63,95],\
                           [0,81,83,85,88,90,92,93,96,0]])
         
    arrayMap["N"]=np.array([[2,0,1,3,4,6,8,10,14,0],\
                           [65,66,33,34,7,9,11,12,16,18],\
                           [67,68,35,36,5,17,13,23,20,22],\
                           [69,70,37,38,48,15,19,25,27,24],\
                           [71,72,39,40,42,50,54,21,29,26],\
                           [73,74,41,43,44,46,52,62,31,28],\
                           [75,76,45,47,51,56,58,60,64,30],\
                           [77,78,82,49,53,55,57,59,61,32],\
                           [79,80,84,86,87,89,91,94,63,0],\
                           [0,81,83,85,88,90,92,93,96,95]])
    
    # mask 0 chanenls
    for key in arrayMap.keys():
         arrayMap[key]=np.ma.array(arrayMap[key],mask=False)
         emptySite=np.where(arrayMap[key]==0)
         for emptyInd in range(4):
                  arrayMap[key].mask[emptySite[0][emptyInd],emptySite[1][emptyInd]]=True          
    
    # move to Python indexing, with the lowest channel index now being 0
    for key in arrayMap.keys():
         arrayMap[key]=arrayMap[key]-1
                   
    if save:         
        targetFolder='../../data/' #or any existing folder where you want to store the output
        filename=targetFolder+'arrayMap.p'
        pickle.dump(arrayMap, open(filename, "wb" ) )
        # arrayMap= pickle.load(open('../../data/arrayMap.p', "rb"))

    return arrayMap