
from scipy.io import loadmat
import numpy as np
import pickle 

#loads and pickle any given dataset

def loadData(sourceFolder,datasetName,targetFolder):

     sourceFile=sourceFolder+datasetName+'.mat'
     FIRA= loadmat(sourceFile)['FIRA'][0].tolist()
     # loadmat(datafile) creates a dictionary with keys dict_keys(['__header__', '__version__', '__globals__', 'FIRA'])
     
     ##################### PYTHONIZE IT ############################
     
     if(FIRA[1].shape[0]!=FIRA[2].shape[0]):
          print('warning: number of trials differs in 2nd and 3rd FIRA field')
     nTrials=min(FIRA[1].shape[0],FIRA[2].shape[0])
     nEvents=FIRA[1].shape[1]
     #anyTrialInd=nTrials-1 # or any number between 0 and nTrials-1
     #nChannels_total =FIRA[2][anyTrialInd,0].shape[0]  #no, must count only until CH_MAX
     CH_MAX = 96; #Number of channels
     
     events=FIRA[0][0][0][6][0].tolist() #tuple, but to be converted into a dictionary
     assert(len(events)==nEvents)
     for eventInd in range(nEvents):
          events[eventInd]=str(events[eventInd][0])
     
     spikingTimes=[]
     values=[]
     
     for trial in range(nTrials):
     
          spikingTimes.append([])
          values.append([])
     
          for event in range(nEvents):
               temp=FIRA[1][trial][event].tolist()
               while type(temp)==list and len(temp)==1:
                    if type(temp[0])==list:
                         temp=temp[0]
                    else:
                         break
               values[trial].append(temp)
               
          for channel in range(CH_MAX):
               spikingTimes[trial].append(FIRA[2][trial,0][channel,0].flatten())
     
     ##################### STITCH TOGETHER THE FAKE TRIALS #####################
               
     # method to get index of trial events
     def IND(eventString):
          ind= [i for i,x in enumerate(events) if x == eventString]
          return(ind[0])
     
     spk_session = [[] for i in range(CH_MAX)] # an empty list of list of length CH_MAX
     stim_times=[]
     stim_durations=[]
     stim_chan=[]
     
     for trl_i in range(nTrials):
     
         #First I calculate the offset for each trial.
         #The offset is calculated as the difference of the refrence
         #time of each trial with respect to the first trial.
         #The refrence point here is the onset of fixation point so I'm
         #also adding the fixation point onset of the first trial to
         #align everything to the beginning of the first trial
     
         trl_offset = values[trl_i][IND('abs_ref')][0]-values[0][IND('abs_ref')][0] - values[0][IND('start_tri')][0]
         
         for ch_i in range (CH_MAX):
              spk_session[ch_i] = np.concatenate((spk_session[ch_i],spikingTimes[trl_i][ch_i]+trl_offset))
     
         if values[trl_i][IND('elestim')]==1:
              stim_times.append(values[trl_i][IND('elestim_on')][0]+trl_offset)
              stim_durations.append(values[trl_i][IND('elestim_off')][0]-values[trl_i][IND('elestim_on')][0])
              stim_chan.append(values[trl_i][IND('ustim_chan')][0])
      
     ## study Spike Counts 
     
     spikeCount=np.empty(CH_MAX)
     for ch_i in range(CH_MAX):
          spikeCount[ch_i]=len(spk_session[ch_i])
     
     ################################### STORING CONVERTED DATA ###########################
     
     ## the following precautional commands aren't needed because we will only analyze hash data. 
     # nUnits=size(spk_session,1);
     # sorting_quality=repmat({'hash'},nUnits,1);
     # unit_id=[[1:nUnits]',zeros(nUnits,1)];
      
     # keys=['spk_session','stim_times','stim_durations','stim_chan','spikeCount']
     dataDict= {}
     dataDict['spk_session']=spk_session
     dataDict['stim_times']=stim_times
     dataDict['stim_durations']=stim_durations
     dataDict['stim_chan']=stim_chan
     dataDict['spikeCount']=spikeCount
     
     targetFile=targetFolder+'spikeData'+datasetName+'.p'
     pickle.dump(dataDict, open(targetFile, "wb" ) )
     # dataDict = pickle.load(open(filename, "rb"))
     
     return()
     
if __name__=="__main__":  
     
     #loading all the datasets

     sourceFolder='../../../SALEH/unsorted/' #or any folder with the content of https://drive.google.com/drive/u/0/folders/12usjkXjnhjhiiRUQAjzM4jnbcqyio8s1
     targetFolder='../../data/' #or any existing folder where you want to store the output
     
     keysLocation='../../data/dataKeys'              
     dataKeys = pickle.load(open(keysLocation, "rb"))
     
     for index,dataset in enumerate(dataKeys):
          print('converting'+str(index+1)+'of'+str(len(dataKeys))+'...')
          loadData(sourceFolder,dataset,targetFolder)

