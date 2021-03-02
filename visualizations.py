# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:40:06 2021

@author: Amin
"""
import matplotlib.pyplot as plt
from scipy import interpolate
import matplotlib as mpl
import numpy as np
import pylab

def visualize_matrix(J,pval=None,titlestr='',fontsize=30,save=False,file=None,cmap='cool'):
    """Visualize a matrix using a pre-specified color map
    
    Args:
        J (numpy.ndarray): 2D (x,y) numpy array of the matrix to be visualized
        pval (numpy.ndarray): a binary matrix with the same size as J corresponding
            to the significane of J's elements; significant elements will be 
            shown by a dot in the middle
        titlestr (string): Title of the plot
        fontsize (integer): Fontsize of the plot
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot
        cmap (matplotlib.cmap): Colormap used in the plot
        
    """
    
    plt.figure(figsize=(10,8))
    if cmap == 'cool': # Used for FCF
        im = plt.imshow(J,cmap=mpl.cm.cool)
    elif cmap == 'copper': # Used for Interventional Connectivity
        im = plt.imshow(J,cmap=mpl.cm.copper)
    elif cmap == 'coolwarm': # Used for Connectome
        im = plt.imshow(J,cmap=mpl.cm.coolwarm)
    
    if pval is not None:
        x = np.linspace(0, pval.shape[0]-1, pval.shape[0])
        y = np.linspace(0, pval.shape[1]-1, pval.shape[1])
        X, Y = np.meshgrid(x, y)
        pylab.scatter(X,Y,s=20*pval, c='k')
    
    plt.axis('off')
    plt.colorbar(im)
    
    plt.xlabel('Neurons',fontsize=fontsize)
    plt.ylabel('Neurons',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    
    if save:
        plt.savefig(file+'.eps',format='eps')
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        pylab.show()
        
def visualize_signals(t, signals, labels, spktimes=None, stim=None, t_range=None, stim_t=None, fontsize=30, save=False, file=None):
    """Visualize a multidimensional signal in time with spikes and a stimulation
        pattern
    
    Args:
        t (numpy.ndarray): 1D numpy array of the time points in which the 
            signals are sampled
        signals (array): An array of multi-dimensional signals where each 
            element is an NxT numpy array; different elements are shown in 
            different subplots
        labels (array): Array of strings where each string is the label of 
            one of the subplots
        spktimes (array): Array of arrays where each element corresponds to
            the spike times of one channel
        stim (array): Stimulation pattern represented as a binary matrix 
            Nxt_stim where N is the number of channels and t_stim is the timing 
            of stimulation
        t_range ((float,float)): Time range used to limit the x-axis
        t_stim (numpy.ndarray): Time points in which the stimulation pattern
            is sampled
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot

    """

    plt.figure(figsize=(10,signals[0].shape[0]/2))
    
    for i in range(len(signals)):
        c = signals[i]
        plt.subplot(1,len(signals),i+1)
        
        plt.title(labels[i],fontsize=fontsize)
        
        offset = np.append(0.0, np.nanmax(c[0:-1,:],1)-np.nanmin(c[0:-1,:],1))
        s = (c-np.nanmin(c,1)[:,None]+np.cumsum(offset,0)[:,None]).T
        
        plt.plot(t, s)
        plt.yticks(s[0,:],[str(signal_idx) for signal_idx in range(s.shape[1])])
        
        if spktimes is not None:
            for k in range(s.shape[1]):
                plt.scatter(spktimes[i][k],np.nanmax(s[:,k])*np.ones((len(spktimes[i][k]),1)),s=10,marker='|')
                
        
        if stim is not None:
            for k in range(stim.shape[1]):
                inp = interpolate.interp1d(stim_t,(stim[:,k]).T,kind='nearest',fill_value='extrapolate',bounds_error=False)
                plt.fill_between(t, t*0+np.nanmin(s[:,k]), t*0+np.nanmax(s[:,k]), where=inp(t)>0,
                                 facecolor='red', alpha=0.1)
        
        if t_range is not None:
            plt.xlim(t_range[0],t_range[1])
        
    if save:
        plt.savefig(file+'.eps',format='eps')
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
