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
