# -*- coding: utf-8 -*-

"""
Created on Tue Aug 11 13:48:10 2020
"""

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.pylab as pyl

def normalize(x,printReport=False):

     """
     normalizing method needed for plotOverMap
     
     """
     z=x.data[x.mask==False]
     dataMin=np.min(z)
     dataMax=np.max(z)
     y=np.zeros(x.data.shape)
     y[x.mask==False]=(x.data[x.mask==False]-dataMin)/(dataMax-dataMin)
     y=np.ma.array(y,mask=False)
     y.mask[x.mask==True]=True
     
     return(y)

   
# %%

def plotOverMap(vecToPlot,nodes_map,sourceNode=None,show=True,printReport=False):
     
     """
     Testing 1
     
     import numpy as np
     import pickle 
     from matplotlib import pyplot as plt 
     arrayMap= pickle.load(open('../../data/arrayMap.p', "rb"))
     x=np.ma.array(np.random.randn(96),mask=False)
     rgba=plotOverMap(x, arrayMap["G"],sourceNode=0,show=True,printReport=False)
     
     
     ###################################################################     

     Testing 2: 
          
     import numpy as np
     import pickle 
     from matplotlib import pyplot as plt 
     arrayMap= pickle.load(open('../../data/arrayMap.p', "rb"))
     x=np.ma.array(np.arange(96),mask=False)
     plotOverMap(x, arrayMap["G"],sourceNode=0)
     plt.show()
     plt.imshow(arrayMap["G"],cmap="summer")

     """          
          
     assert( len(vecToPlot)-1==np.max(nodes_map))

     # channels where arrayToPlot=0 should be WHITE
     # The channel labeled as "source" (number in matlab indexing convention) should be RED
     # the other channels should go from YELLOW to BLUE. 
     
     #The default colors of plt.imshow are blue to yellow (going through green with no red and no white)
     matrixToPlot=vecToPlot[nodes_map]
     matrixToPlot=np.ma.array(matrixToPlot,mask=False)
     matrixToPlot.mask[nodes_map.mask==True]=True
     
     if sourceNode!=None:
          sourceSite=np.where(nodes_map==sourceNode)
          matrixToPlot.mask[sourceSite[0], sourceSite[1]]=True     
     
     cmap = plt.cm.summer

     if printReport==True:
          print("before normalization")
          print(matrixToPlot)
     matrixToPlot=normalize(matrixToPlot,printReport=True)

     if printReport==True:
          print("after normalization")
          print(matrixToPlot)
     
     rgba = cmap(matrixToPlot)
     
     if printReport==True:
          print("rgba=")
          print(rgba)    
          
     rgba=np.ma.array(rgba,mask=False)
     rgba.mask[nodes_map.mask==True]=True
               
     if sourceNode!=None:
          sourceSite=np.where(nodes_map==sourceNode)
          rgba.data[sourceSite[0], sourceSite[1], :4] = 1, 0, 0, 1
          rgba.mask[sourceSite[0], sourceSite[1], :4] = False
          
     if show==True:
          plt.imshow(rgba, interpolation='nearest')
          plt.show()
          plt.close('all')
      
     
     return(rgba)

        
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
