# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:42:55 2020

"""

nBinsToPlot=25
colors = pyl.cm.brg(np.linspace(0,1,nBinsToPlot))
incrs=output["incrementsByLapse"] # incrs.shape = (n_postBins,nEvents,nChannels,)

nChannelsToPlot=96
plt.title("rate increment")

channel=14
counts,edges=np.histogram(incrs[lapseInd ,:,channel])
          midpoints=(edges[1:]+edges[:-1])/2
          plt.plot(midpoints,counts, color=colors[lapseInd], label=str(output["postBinEdges"][lapseInd+1])+" ms")
          #plt.legend()
     plt.show()
    
