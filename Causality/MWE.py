# %%

from scipy import stats
import matplotlib.pyplot as plt 
import matplotlib.pylab as pyl

plotInd=0

resp_measures=responseOutputs[plotInd][0][resp_measure_name]
causalPower=causalPowers[plotInd]
lapses=responseOutputs[plotInd][1]['lapses']
stimulatedCh=stimulatedChs[plotInd]

(nLapses,nChannels)=resp_measures.shape
assert(len(lapses)==nLapses)
assert(nChannels==len(causalPower))     
corrcoefs=np.zeros(nLapses)
pValues=np.zeros(nLapses)
for lapseInd in range(nLapses):
     unmasked=~resp_measures.mask[lapseInd,:]
     corrcoefs[lapseInd],pValues[lapseInd]=stats.pearsonr(resp_measures.data[lapseInd,:][unmasked], causalPower[unmasked])

fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(lapses,corrcoefs)
ax1.set_title("stimulated channel"+str(stimulatedCh+1))           
ax1.set_xlabel("post-stimulus lapse (ms)")
ax1.set_ylabel("corr_coef")

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(lapses,np.log(pValues))
ax2.set_title("stim_ch="+str(stimulatedCh+1))           
xmin,xmax= ax1.get_xlim() 
plt.hlines(np.log(5*10**(-2)), xmin, xmax,colors='r')
ax2.set_xlabel("post-stimulus lapse (ms)")
ax2.set_ylabel("log(p-value)")

# show plots
fig.tight_layout()
figureName= "../../data/"+"stim_ch="+str(stimulatedCh+1)+"resp_measure_name+", 
plt.savefig(figureName, bbox_inches='tight')

fig.show()
