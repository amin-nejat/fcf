# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 17:40:06 2021

@author: Amin
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import pylab

from scipy.ndimage import gaussian_filter
from scipy import interpolate
from scipy import stats

import numpy as np

from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

import networkx as nx
# %%
def visualize_matrix(J,pval=None,titlestr='',fontsize=30,save=False,file=None,cmap='cool'):
    '''Visualize a matrix using a pre-specified color map
    
    Args:
        J (numpy.ndarray): 2D (x,y) numpy array of the matrix to be visualized
        pval (numpy.ndarray): a binary matrix with the same size as J corresponding to the significane of J's elements; significant elements will be shown by a dot in the middle
        titlestr (string): Title of the plot
        fontsize (integer): Fontsize of the plot
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot
        cmap (matplotlib.cmap): Colormap used in the plot: cool, copper, coolwarm
    '''
    
    plt.figure(figsize=(10,8))
    im = plt.imshow(J,cmap=cmap)
    
    if pval is not None:
        x = np.linspace(0, pval.shape[0]-1, pval.shape[0])
        y = np.linspace(0, pval.shape[1]-1, pval.shape[1])
        X, Y = np.meshgrid(x, y)
        pylab.scatter(X,Y,s=20*pval, c='k')
    
    plt.axis('off')
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=fontsize)
    
    plt.xlabel('Neurons',fontsize=fontsize)
    plt.ylabel('Neurons',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        pylab.show()
        
# %%
def visualize_signals(t, signals, labels, spktimes=None, stim=None, t_range=None, stim_t=None, fontsize=20, save=False, file=None):
    '''Visualize a multidimensional signal in time with spikes and a stimulation pattern
    
    Args:
        t (np.ndarray): 1D numpy array of the time points in which the signals are sampled
        signals (array): An array of multi-dimensional signals where each element is an NxT numpy array; different elements are shown in different subplots
        labels (array): Array of strings where each string is the label of one of the subplots
        spktimes (array): Array of arrays where each element corresponds to the spike times of one channel
        stim (array): Stimulation pattern represented as a binary matrix Nxt_stim where N is the number of channels and t_stim is the timing of stimulation
        t_range ((float,float)): Time range used to limit the x-axis
        t_stim (numpy.ndarray): Time points in which the stimulation pattern is sampled
        save (bool): Whether to save the plot or not
        file (string): File address to save the plot
    '''

    plt.figure(figsize=(10,1.4*signals[0].shape[0]))
    
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
            
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()


# %%
def show_clustered_connectivity(adjacency,clusters,exc,save=False,file=None):
    '''Visualize clustered connectivity graph
        
    Args:
        adjacency (matrix): Adjacency matrix of the connectivity
        clusters (float): Array of cluster sizes
        exc (integer): Number of excitatory nodes for coloring
        save (bool): If True the plot will be saved
        file (string): File address for saving the plot
    '''
    
    G = nx.from_numpy_array(adjacency,create_using=nx.DiGraph)
    weights = nx.get_edge_attributes(G,'weight').values()
    
    G_ = nx.from_numpy_array(np.ones((len(clusters),len(clusters))))
    pos = np.array(list(nx.spring_layout(G_, iterations=100).values()))
    pos = np.repeat(pos, clusters, axis=0)        
    
    rpos = np.hstack([np.array([.08*np.cos(np.linspace(0,2*np.pi,1+clusters[i])[:-1]), 
                                .08*np.sin(np.linspace(0,2*np.pi,1+clusters[i])[:-1])]) 
            for i in range(len(clusters))])
            
    plt.figure(figsize=(10,10))
    
    node_color = np.array([[0,0,1,.5]]*exc + [[1,0,0,.5]]*(G.number_of_nodes()-exc))
    
    options = {
        'node_color': node_color,
        'edgecolors': 'k',
        'node_size': 300,
        'width': 2*np.array(list(weights)),
        'arrowstyle': '-|>',
        'arrowsize': 15,
        'font_size':10, 
        'font_family':'fantasy',
        'connectionstyle':"arc3,rad=-0.1",
    }
    
    nx.draw(G, pos=list(pos+rpos.T), with_labels=True, arrows=True, **options)
    
    if save:
        plt.savefig(file+'.pdf',format='pdf')
        plt.savefig(file+'.png',format='png')
        plt.close('all')
    else:
        plt.show()
        
# %%
def show_downstream_connectivity(adjacency,fontsize=20,save=False,file=None):
    '''Visualize downstream connectivity graph
        
    Args:
        adjacency (matrix): Adjacency matrix of the connectivity
        fontsize (float): Font size used for plotting
        save (bool): If True the plot will be saved
        file (string): File address for saving the plot
    '''
    
    G = nx.from_numpy_array(adjacency,create_using=nx.DiGraph)
    weights = nx.get_edge_attributes(G,'weight').values()
    
    node_color = np.array([[0,0,1,.5]]*3 + [[1,0,1,.5]]*(G.number_of_nodes()-3))
    
    if adjacency.shape[0] == 10:
        options = {
            'node_color': node_color,
            'edgecolors': 'k',
            'node_size': 3000,
            'width': 20*np.array(list(weights)),
            'arrowstyle': '-|>',
            'arrowsize': 20,
            'font_size':fontsize, 
            'font_family':'fantasy',
            'connectionstyle':'arc3,rad=0',
        }
        plt.figure(figsize=(10,16))
    elif adjacency.shape[0] > 100:
        node_size = np.concatenate((np.ones((3,1)),np.zeros((G.number_of_nodes()-3,1))))
        options = {
            'node_color': node_color,
            'edgecolors': 'k',
            'node_size': node_size*2500+500,
            'width': 1*np.array(list(weights)),
            'arrowstyle': '-|>',
            'arrowsize': 20,
            'font_size':fontsize, 
            'font_family':'fantasy',
            'connectionstyle':'arc3,rad=0',
        }
        plt.figure(figsize=(10,6))

    pos = nx.bipartite_layout(G,set(np.arange(3)),align='horizontal')
    
    pos = np.array(list(pos.values()))
    
    m1 = pos[:3,:].mean(0)
    m2 = pos[3:,:].mean(0)
    
    pos[:3,:] = m1[None,:] + .1*np.array([np.sin(np.linspace(0,2*np.pi,4)[:-1]), 
                                       np.cos(np.linspace(0,2*np.pi,4)[:-1])]).T
            
    pos[3:,:] = m2[None,:] + .2*np.array([np.sin(np.linspace(0,2*np.pi,G.number_of_nodes()-2)[:-1]), 
                                       np.cos(np.linspace(0,2*np.pi,G.number_of_nodes()-2)[:-1])]).T
    
    
    nx.draw(G, pos=pos, with_labels=True, arrows=True, **options)
    
    if save:
        plt.savefig(file+'.pdf',format='pdf')
        plt.savefig(file+'.png',format='png')
        plt.close('all')
    else:
        plt.show()
        
# %%
def visualize_nx_graph(G,save=False,file=None):
    '''calling networkx draw function on an input graph for visualization
    '''
    nx.draw(G, with_labels=True)
    if save:
        plt.savefig(file+'.pdf',format='pdf')
        plt.savefig(file+'.png',format='png')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_EI(J,E,I,X,save=False,file=None):
    '''visualize excitatory-inhibitory connectivity graph
    '''
    plt.figure(figsize=(10,10))

    node_color = np.array([[0,0,1,.5]]*E + [[1,0,0,.5]]*I)
    G = nx.from_numpy_array(J,create_using=nx.DiGraph)
    weights = nx.get_edge_attributes(G,'weight').values()
    
    options = {
        'node_color': node_color,
        'edgecolors': 'k',
        'node_size': 300,
        'width': 2*np.array(list(weights)),
        'arrowstyle': '-|>',
        'arrowsize': 15,
        'font_size':10, 
        'font_family':'fantasy',
        'connectionstyle':"arc3,rad=-0.1",
    }

    nx.draw(G, pos=list(X), with_labels=True, arrows=True, **options)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_state(states, pars=None, titlestr='', fontsize=30, save=False, file=None, smooth=False):
    '''visualize multivariate time series in the sate space
    '''
    plt.figure(figsize=(10,10))
    
    if smooth:
        states = np.array([gaussian_filter(states[:,i],2) for i in range(states.shape[1])]).T

    if states.shape[1] == 2:
        plt.plot(states[:,0], states[:,1], '-k', lw=1)
    else:
        plt.subplot(111, projection='3d')
        
        pca = PCA(n_components=3)
        pca.fit_transform(states)
        states_pca = pca.transform(states)
        plt.plot(states_pca[:,0], states_pca[:,1], states_pca[:,2], '-k', lw=2)
        
    plt.xlabel("$x_1$",fontsize=fontsize)
    plt.ylabel("$x_2$",fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr,fontsize=fontsize)
    plt.tight_layout()
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
        
        
# %%
def visualize_cnn(J,pval=None,titlestr='',fontsize=30,save=False,file=None,cmap='cool',newfig=True,show=True):
    '''visualize (effective) connectivity given a colormap
    '''
    if newfig: plt.figure(figsize=(10,8))
    im = plt.imshow(J,cmap=cmap)
    
    if pval is not None:
        x = np.linspace(0, pval.shape[0]-1, pval.shape[0])
        y = np.linspace(0, pval.shape[1]-1, pval.shape[1])
        X, Y = np.meshgrid(x, y)
        pylab.scatter(X,Y,s=20*pval,c='k')
    
    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=fontsize)
    
    plt.xlabel('Neurons',fontsize=fontsize)
    plt.ylabel('Neurons',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.title(titlestr.upper(),fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_stim_protocol(I,time_st,time_en,N,fontsize=10,save=False,file=None):
    '''showing stimulation protocol on an image with 1's for stimulation and 0's for rest
    '''
    plt.figure()
    plt.imshow(I.T,aspect='auto',interpolation='none', extent=[time_st,time_en,0,N],origin='lower')
    plt.xlabel('time',fontsize=fontsize)
    plt.ylabel('Neurons',fontsize=fontsize)
    plt.title('Stimulation Protocol',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def visualize_spikes(spktimes,labels,stim=None,stim_t=None,time=None,fontsize=30,save=False,file='',t_range=None,distinct_colors=False,distinction_point=0):
    '''visualize spike raster
    '''
    plt.figure(figsize=(10,5))
    
    for i in range(len(spktimes)):
        plt.subplot(1,len(spktimes),i+1)
        spk = np.array(spktimes[i])
        
        if distinct_colors:
            plt.scatter(spk[spk[:,0]>=distinction_point,1],spk[spk[:,0]>=distinction_point,0],s=10,color='r',marker='|')
            plt.scatter(spk[spk[:,0]< distinction_point,1],spk[spk[:,0]< distinction_point,0],s=10,color='b',marker='|')
        else:
            plt.scatter(spk[:,1],spk[:,0],s=10,color='k',marker='|')
        
        plt.title(labels[i],fontsize=fontsize)
        
        plt.xlabel('Time',fontsize=fontsize)
        plt.ylabel('Neuron',fontsize=fontsize)
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        
        if stim is not None:
            for k in range(spk[:,0].max().astype(int)):
                inp = interpolate.interp1d(stim_t,(stim[:,k]).T,kind='nearest',fill_value='extrapolate',bounds_error=False)
                plt.fill_between(time, time*0+k, time*0+k+1, where=inp(time)>0,
                                 facecolor='red', alpha=0.1)
                
        plt.ylim((0,max(spk[:,0])))
        if t_range is not None:
            plt.xlim(t_range)
            
        
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_scatters(x1,x2,sig,xlabel='',ylabel='',titlestr='',fontsize=30,save=False,file=None):
    '''visualize multiple scatter plots using subplots
    '''
    plt.figure(figsize=(len(x1)*10,10))
    for i in range(len(x1)):
        plt.subplot(1,len(x1),i+1)
        visualize_scatter(
                x1[i][~np.isnan(x1[i]+x2[i])],x2[i][~np.isnan(x1[i]+x2[i])],sig[i][~np.isnan(x1[i]+x2[i])],
                xlabel=xlabel[i],ylabel=ylabel[i],titlestr=titlestr,
                fontsize=fontsize,openfig=False
            )
            
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
# %%
def visualize_scatter(x1,x2,sig=None,xlabel='',ylabel='',titlestr='',fontsize=30,openfig=True,save=False,file=None):
    '''visualize a scatter plot and fitting a line and showing correlations
    '''
    if openfig: plt.figure(figsize=(10,10))
    plt.scatter(x1,x2,s=20,c='k')
    
    if sig is not None:
        plt.scatter(x1,x2,s=sig*30,edgecolors='r',facecolors='none')
        
    plt.xlabel(xlabel.upper(),fontsize=fontsize)
    plt.ylabel(ylabel.upper(),fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    try:
        m, b = np.polyfit(x1,x2,1)
        plt.plot(x1,m*x1+b,'k',fontsize=fontsize)
    except:
        pass
    plt.grid('on')
    
    try:
        scorr,spval = stats.spearmanr(x1,x2)
        pcorr,ppval = stats.pearsonr(x1,x2)
        
        
        str_ = 'SP(' + '{:.2f}'.format(scorr) + ',' + '{:.1e}'.format(spval) + ')\n'\
               'PE(' + '{:.2f}'.format(pcorr) + ',' + '{:.1e}'.format(ppval) + ')'
        
        plt.annotate(str_, xy=(.02,.95), xycoords='axes fraction',
                size=14, ha='left', va='top',
                bbox=dict(boxstyle='round', fc='w'),fontsize=fontsize)
    except:
        pass
    
    plt.title(titlestr,fontsize=fontsize)

    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
        
# %%
def visualize_adjacency(adjacency,fontsize=20,save=False,file=''):
    '''visualize network adjancency matrix
    '''
    plt.figure(figsize=(10,10))
    cmap = mpl.cm.RdBu
    plt.imshow(adjacency,cmap=cmap)
    plt.xlabel('Neurons',fontsize=fontsize)
    plt.ylabel('Neuron',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=fontsize)

    plt.clim([-abs(adjacency.max()/2),abs(adjacency.max()/2)])
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()

# %%
def visualize_cnn_physical_layout(layout,cnn,pval=None,cmap='cool',titlestr='',fontsize=30,save=False,file=None):
    '''visualize connectivity on an electroed array physical layout
    '''
    def gini(x):
        x = x-np.nanmin(x)+1e-10
        mad = np.abs(np.subtract.outer(x, x)).mean()
        rmad = mad/np.mean(x)
        g = 0.5 * rmad
        return g    
    
    plt.figure(figsize=(10,10))
    
    cnn_reshaped = cnn[layout]
    cnn_reshaped[layout.mask] = np.nan
    
    im = plt.imshow(cnn_reshaped,cmap=cmap)
        
    if pval is not None:
        pval_reshaped = pval[layout]
        pval_reshaped[layout.mask] = False
        x = np.linspace(0, pval_reshaped.shape[0]-1, pval_reshaped.shape[0])
        y = np.linspace(0, pval_reshaped.shape[1]-1, pval_reshaped.shape[1])
        X, Y = np.meshgrid(x, y)
        pylab.scatter(X,Y,s=20*pval_reshaped, c='k')

    cbar = plt.colorbar(im)
    cbar.ax.tick_params(labelsize=fontsize)
    plt.title(titlestr.upper()+', gini:'+'{:.2f}'.format(gini(cnn[~np.isnan(cnn)])),fontsize=fontsize)
    plt.axis('off')
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def visualize_bar(cnn,sig,titlestr='',fontsize=30,openfig=False,save=False,file=None):
    '''visualize bar plot of significant vs. non-significant functional connectivity
    according to the pvalues of interventional connectivity (or vice versa)
    '''
    if openfig: plt.figure(figsize=(10,20))
    plt.bar(
        np.arange(2),height=[cnn[~sig].mean(),cnn[sig].mean()],
        width=.3,yerr=[stats.sem(cnn[~sig]),stats.sem(cnn[sig])],
        color=[0,0,0,.5],edgecolor='k',tick_label=['Upstream', 'Downstram']
    )
    
    _,p = stats.ttest_ind(cnn[~sig],cnn[sig])
    plt.title(titlestr.upper()+' p-value:' + '{:.1e}'.format(p),fontsize=fontsize)
    
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
        
# %%
def visualize_bars(cnn,sig,titlestr='',fontsize=30,save=False,file=None):
    '''visualize multiple scatter plots using subplots
    '''
    plt.figure(figsize=(len(cnn)*10,10))
    for i in range(len(cnn)):
        plt.subplot(1,len(cnn),i+1)
        visualize_bar(cnn[i][~np.isnan(cnn[i])],sig[i][~np.isnan(cnn[i])],titlestr[i],fontsize=fontsize,openfig=False)
            
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()
        
# %%
def plot_index_vs_distance(array,ccms,titlestr='',fontsize=30,save=False,file=None):
    '''Plot connectivity index as a function of physical distance of the source channel
    '''
    plt.figure(figsize=(8,5))
    
    bins = np.linspace(0,13,14)
    
    ss = []
    for ccm in ccms:
        for i in range(ccm.shape[1]):
            v = ccm[:,i]
            v[i] = np.nan
            v_reshaped = v[array]
            v_reshaped[array.mask] = np.nan
            
            ch_loc = np.array(np.where(array == i))
            
            dist_cnn = np.hstack((
                v_reshaped.flatten()[:,None],
                cdist(np.array(np.where(np.ones(v_reshaped.shape))).T,ch_loc.T)
            ))
            ss.append(dist_cnn[~np.isnan(dist_cnn.sum(1)),:])
            
    
    ss = np.vstack(ss)
    ss = np.array(ss)
    
    s_mean,edges,_ = stats.binned_statistic(ss[:,1],ss[:,0],statistic='median',bins=bins)
    s_serm,edges,_ = stats.binned_statistic(ss[:,1],ss[:,0],statistic=stats.sem,bins=bins)
    
    plt.plot(edges[:-1]+np.diff(edges)/2, s_mean, c='k')
    plt.fill_between(edges[:-1]+np.diff(edges)/2, s_mean-s_serm, s_mean+s_serm,alpha=.2)
    plt.grid('on')
    
    plt.xlim(0,9)
    
    plt.xlabel('Distance',fontsize=fontsize)
    plt.ylabel('FCF',fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    
    if save:
        plt.savefig(file+'.png',format='png')
        plt.savefig(file+'.pdf',format='pdf')
        plt.close('all')
    else:
        plt.show()