#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:49:24 2020

@author: bagger
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy.stats import norm, chi2
import numpy.linalg as linalg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from scipy.interpolate import splrep, splev
from scipy.optimize import minimize
from matplotlib.ticker import MultipleLocator
import csv
from numpy import genfromtxt
import pandas as pd
#from pandas.tools.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.artist import setp
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
from os import path
import ase.db
from ase.db import connect
from matplotlib.legend_handler import HandlerTuple
from ase.io import write, read
from ase.visualize import view
from scipy.stats import norm
### large data import
import re
r = re.compile("([a-zA-Z]+)([0-9]+)")


L=3; size1=28; size2=20; size3=24;A1=0.5;
s1=15; s2=15; s3=15; sdot=150; sdot2=150;
size1=28;size2=20; sdot=150

def seperate_string_number(string):
    previous_character = string[0]
    groups = []
    newword = string[0]
    for x, i in enumerate(string[1:]):
        if i.isalpha() and previous_character.isalpha():
            newword += i
        elif i.isnumeric() and previous_character.isnumeric():
            newword += i
        else:
            groups.append(newword)
            newword = i

        previous_character = i

        if x == len(string) - 2:
            groups.append(newword)
            newword = ''
    return groups


#############################################################
#### We drop some data that are strong outliers in the dataset & save it
##################################################################


df=pd.read_csv('dataset.csv')  
#df.plot(x='V SHE', y='NH3_reaction', kind='scatter')


## 
# Checking for outliers
##
Features=['Nitride Formation Energy', 'Hydride Formation Energy',
       'Oxide Formation Energy', 'MxOyHz Formation Energy',
       'MxCyOz Formation Energy', 'MxFy Formation Energy', 'MxOyHz Reaction',
       'MxOy Reaction', 'MxHy Reaction', 'Mx Reaction', 'MN3 Formation Energy',
       'M3N Formation Energy', 'N_energy', 'N2_energy', 'NH2_energy',
       'NH3_energy']

for feat in Features:
    lower_limit = df[feat].mean() - 2*df[feat].std()
    upper_limit = df[feat].mean() + 2*df[feat].std()

    df_filtered=df[(df[feat]>lower_limit)&(df[feat]<upper_limit)]
    df_removed=df[(df[feat]<lower_limit) | (df[feat]>upper_limit)]
    print(feat + ':   ' + str(df_filtered.shape))
    print(df_removed['Mname'])



#############################################################
#### defining features
##################################################################

Features=['Nitride Formation Energy', 'Hydride Formation Energy',
       'Oxide Formation Energy', 'MxOyHz Formation Energy',
       'MxCyOz Formation Energy', 'MxFy Formation Energy', 'MxOyHz Reaction',
       'MxOy Reaction', 'MxHy Reaction', 'Mx Reaction', 'MN3 Formation Energy',
       'M3N Formation Energy', 'N_energy', 'N2_energy', 'NH2_energy',
       'NH3_energy']

Features_label=['$\Delta$H$_{Nitride}$', '$\Delta$H$_{Hydride}$',
       '$\Delta$H$_{Oxide}$', '$\Delta$H$_{M_xO_yH_z}$','$\Delta$H$_{M_xC_yO_z}$','$\Delta$H$_{M_xF_y}$',
        '$\Delta$E$_{M_xO_yH_z}$','$\Delta$E$_{M_xO_y}$','$\Delta$E$_{M_xH_y}$', '$\Delta$E$_{M_x}$',
        '$\Delta$H$_{MN_3}$','$\Delta$H$_{M_3N}$','$\Delta$E$_{~^*\!N}$', 
        '$\Delta$E$_{~^*\!N_2}$', '$\Delta$E$_{~^*\!NH_2}$',
       '$\Delta$E$_{~^*\!NH_3}$']

Features_small=['Nitride Formation Energy', 'Hydride Formation Energy',
       'Oxide Formation Energy', 'MxOyHz Formation Energy',
       'MxCyOz Formation Energy', 'MxFy Formation Energy', 'MN3 Formation Energy',
       'M3N Formation Energy', 'N_energy', 'N2_energy', 'NH2_energy',
       'NH3_energy']

Features_small_label=['$\Delta$H$_{Nitride}$', '$\Delta$H$_{Hydride}$',
       '$\Delta$H$_{Oxide}$', '$\Delta$H$_{M_xO_yH_z}$','$\Delta$H$_{M_xC_yO_z}$','$\Delta$H$_{M_xF_y}$',
        '$\Delta$H$_{MN_3}$','$\Delta$H$_{M_3N}$','$\Delta$E$_{~^*\!N}$', 
        '$\Delta$E$_{~^*\!N_2}$', '$\Delta$E$_{~^*\!NH_2}$',
       '$\Delta$E$_{~^*\!NH_3}$']


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################


df['Dist to Li']=0.0
for feat in Features_small:
    df['Dist to Li']=df['Dist to Li']+(df[feat]-df.loc[df['Mname']=='Li'][feat].values)**2
df['Dist to Li']=df['Dist to Li']**(1/2)

df_small=df[['Mname', 'Dist to Li']].dropna()
df_small.to_csv('your.csv', index=False, header=False)


# # # https://github.com/arosen93/ptable_trends
from ptable_trends import ptable_plotter
ptable_plotter("your.csv", cmap="plasma", alpha=1.0, extended=False)



fig = plt.figure(figsize=(14,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)

for k in range(0,len(df['Dist to Li'].values)):
    if df['Mname'].values[k]=='Li' or df['Mname'].values[k]=='Nb' or df['Mname'].values[k]=='Y' or df['Mname'].values[k]=='Sc':
        print('not Li or Nb or Y or Sc')
    
    else:
        if df['Mname'].values[k]=='Ca' or df['Mname'].values[k]=='K':
            plt.text(df['V SHE'].values[k], df['Dist to Li'].values[k]-1, df['Mname'].values[k], fontsize=size1)            
        elif df['Mname'].values[k]=='Ba' or df['Mname'].values[k]=='Cs':
            plt.text(df['V SHE'].values[k]-0.25, df['Dist to Li'].values[k], df['Mname'].values[k], fontsize=size1)            
        else:
            plt.text(df['V SHE'].values[k], df['Dist to Li'].values[k], df['Mname'].values[k], fontsize=size1)
    
        if df['Dissociation'].values[k]==True:
            p1=plt.scatter(df['V SHE'].values[k], df['Dist to Li'].values[k], c='b', s=sdot)
        elif df['Dissociation'].values[k]==False:
            p2=plt.scatter(df['V SHE'].values[k], df['Dist to Li'].values[k], c='r', s=sdot)

plt.plot([-3,1],[3,13],'k', alpha=0.3, linewidth=4)
plt.xlabel(r'Standard Reduction Potential [V vs SHE]',fontsize=size1)
plt.ylabel(r'D$_{Li}$=$\sqrt{\sum_{f} ( E_f^{m} - E_f^{Li})^2}$',fontsize=size1)
plt.xlim([-3.2,1.55])
plt.ylim([0,13])
l = ax.legend([p1,p2],['Cleaved phase','Coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0)
fig.text(0.02, 0.9, '(b)', ha='center', fontsize=size2+10)
plt.savefig('D_to_Li.png', dpi=400, bbox_inches='tight')

df['Dist to Ca']=0.0
for feat in Features_small:
    df['Dist to Ca']=df['Dist to Ca']+(df[feat]-df.loc[df['Mname']=='Ca'][feat].values)**2
df['Dist to Ca']=df['Dist to Ca']**(1/2)


fig = plt.figure(figsize=(14,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)

for k in range(0,len(df['Dist to Ca'].values)):
    if df['Mname'].values[k]=='Ca' or df['Mname'].values[k]=='Nb' or df['Mname'].values[k]=='Y' or df['Mname'].values[k]=='Sc':
        print('not Ca or Nb or Y or Sc')
    
    else:
        plt.text(df['V SHE'].values[k], df['Dist to Ca'].values[k], df['Mname'].values[k], fontsize=size1)
    
        if df['Dissociation'].values[k]==True:
            p1=plt.scatter(df['V SHE'].values[k], df['Dist to Ca'].values[k], c='b', s=sdot)
        elif df['Dissociation'].values[k]==False:
            p2=plt.scatter(df['V SHE'].values[k], df['Dist to Ca'].values[k], c='r', s=sdot)
    
plt.xlabel(r'Standard Reduction Potential [V vs SHE]',fontsize=size1)
#plt.ylabel(r'D$_{Ca}$=($\sum_{f}$ ( E$_f^{m}$ - E$_f^{Ca}$)$^2$',fontsize=size1)

plt.ylabel(r'D$_{Ca}$=$\sqrt{\sum_{f} ( E_f^{m} - E_f^{Ca})^2}$',fontsize=size1)
plt.xlim([-3.2,1.55])
plt.ylim([0,12.5])
l = ax.legend([p1,p2],['Cleaved phase','Coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0)
plt.savefig('D_to_Ca.png', dpi=400, bbox_inches='tight')

#############################################################
#### Calculating Z-scores
##################################################################


for feat in Features:
    df['Z_score_'+feat]=(df[feat]-df[feat].mean())/df[feat].std()


#############################################################
#### plotting of figures  All vs V SHE
#############################################################

for yval, ylabel in zip(Features,Features_label):
    fig = plt.figure(figsize=(14,7));
    ax=fig.gca()
    plt.locator_params(axis='x',nbins=5);plt.grid(True)
    plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
    
    hist,bin_edges = np.histogram(df[yval].dropna().values)
    plt.barh(bin_edges[:-1], (hist)/hist.max(),left=-3.2, height = bin_edges[1]-bin_edges[0], color='#0504aa',alpha=0.2)
    
    ymin=np.sort(df[yval].dropna().values).min()-(bin_edges[1]-bin_edges[0])
    ymax=np.sort(df[yval].dropna().values).max()+(bin_edges[1]-bin_edges[0])
    values=np.linspace(ymin,ymax,1000)
    
    plt.plot(-3.2+norm.pdf(values,df[yval].dropna().mean(),df[yval].dropna().std()),values)
    plt.plot([-5,5],[df.loc[df['Mname']=='Li'][yval].values,df.loc[df['Mname']=='Li'][yval].values],'k', alpha=0.3, linewidth=4)
    plt.plot([-5,5],[df.loc[df['Mname']=='Ca'][yval].values,df.loc[df['Mname']=='Ca'][yval].values],'k', alpha=0.3, linewidth=4)
    #plt.text(-2.9,df.loc[df['Mname']=='Li'][yval].values,r'Z$_{Li}$='+str(np.round(df.loc[df['Mname']=='Li']['Z_score_'+yval].values[0],2)),fontsize=size1)
    plt.text(-2.9,max(df[yval].values),r'Z$_{Li}$='+str(np.round(df.loc[df['Mname']=='Li']['Z_score_'+yval].values[0],2))+ ',  Z$_{Ca}$='+str(np.round(df.loc[df['Mname']=='Ca']['Z_score_'+yval].values[0],2)),fontsize=size1)
     
    
    for k in range(0,len(df[yval].values)):
        if df['Mname'].values[k]=='Au':
            plt.text(df['V SHE'].values[k]-0.2, df[yval].values[k], df['Mname'].values[k], fontsize=size1)        
        else:
            plt.text(df['V SHE'].values[k], df[yval].values[k], df['Mname'].values[k], fontsize=size1)
    
        if df['Dissociation'].values[k]==True:
            p1=plt.scatter(df['V SHE'].values[k], df[yval].values[k], c='b', s=sdot)
        elif df['Dissociation'].values[k]==False:
            p2=plt.scatter(df['V SHE'].values[k], df[yval].values[k], c='r', s=sdot)


    
    plt.xlabel(r'Standard Reduction Potential [V vs SHE]',fontsize=size1)
    plt.ylabel(ylabel +' [eV]',fontsize=size1)
    plt.xlim([-3.2,1.55])
    l = ax.legend([p1,p2],['Cleaved phase','Coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0)
    #plt.savefig(yval+'_vs_SHE.png', dpi=400, bbox_inches='tight')
    if yval=='N_energy':
        fig.text(0.02, 0.9, '(a)', ha='center', fontsize=size2+10)
    
    plt.savefig(yval+'_vs_SHE.png', dpi=400, bbox_inches='tight')