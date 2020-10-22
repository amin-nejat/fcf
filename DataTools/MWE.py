# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 20:40:34 2020

"""


from scipy.io import loadmat 
import pickle 
from dataLoader_standardFormat import convertData
 

#loading all the datasets

spikeSorting=True
if spikeSorting==False:
     sourceFolder='../../../SALEH/unsorted/' #or any folder with the content of https://drive.google.com/drive/u/0/folders/12usjkXjnhjhiiRUQAjzM4jnbcqyio8s1
     keysLocation='../../DATA/dataKeys'
     dataKeys = pickle.load(open(keysLocation, "rb"))
if spikeSorting==True:
     sourceFolder='../../../SALEH/sorted/' #or any folder with the content of https://drive.google.com/drive/u/0/folders/12usjkXjnhjhiiRUQAjzM4jnbcqyio8s1
     #dataKeys=[["G20191106a.FIRA.resting.PAG","G20191106b.FIRA.resting.ustim.PAG"],["G20191107a.FIRA.resting.PAG", "G20191107b.FIRA.resting.ustim.PAG"],["G20191108a.FIRA.resting.PAG","G20191108b.FIRA.resting.ustim.PAG"]]
     dataKeys=[[0,"G20191106b.FIRA.resting.ustim.PAG"]]

targetFolder='../../DATA/' #or any existing folder where you want to store the output
dataTypes=["resting","stimulated"]
for dataIndex in range(len(dataKeys)):
     print('converting dataset '+str(dataIndex+1)+' of '+str(len(dataKeys))+'...')
     for dataType in range(2):
          if dataKeys[dataIndex][dataType]!=0:
               datasetName=dataKeys[dataIndex][dataType]
               sourceFile=sourceFolder+datasetName+'.mat'
               FIRA= loadmat(sourceFile)['FIRA'][0].tolist()
               dataDict=convertData(FIRA)
               dataDict["dataset_ID"]=datasetName
               dataDict["data_type"]=dataTypes[dataType]
               targetFile=targetFolder+'session'+str(dataIndex)+"_part"+str(dataType+1)+'.p'
               pickle.dump(dataDict, open(targetFile, "wb" ))
               # dataDict = pickle.load(open(targetFile, "rb"))
               if spikeSorting==True:
                    pickle.dump(dataDict, open(targetFolder+datasetName+".p", "wb" ))
