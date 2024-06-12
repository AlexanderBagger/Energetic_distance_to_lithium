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



#############################################################
#### K point and cutoff convergence
##################################################################


Kpoint_Li=[1,2,3,4,5,6,7,8,9,10]
PW_Li=[300,400,500,600,700,800]

E_Kpoint_Li=[-3.8925,-0.7105,-0.9125,-0.7135,-0.7405,-0.7025,-0.7135,-0.6975,-0.6975,-0.6925]
E_PW_Li=[-1.0445,-0.765,-0.7375,-0.7415,-0.7605,-0.76]


fig = plt.figure(figsize=(7,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
plt.scatter(Kpoint_Li, E_Kpoint_Li, c='b', label='Li @ Convergence check')
plt.plot(Kpoint_Li, E_Kpoint_Li,'b')
Kpoint_Li_DFT=4
plt.plot([Kpoint_Li_DFT,Kpoint_Li_DFT],[np.min(E_Kpoint_Li),np.max(E_Kpoint_Li)],'k', label='Used value')
#plt.plot([3.4,3.4],[np.min(E_N_Li),np.max(E_N_Li)])
plt.xlabel(r'K-points',fontsize=size1)
plt.ylabel(r'$\Delta$E$_{~^*\!N}$ [eV]',fontsize=size1)
plt.legend(loc=1, fontsize=size2)
plt.ylim([-1.0,-0.5])
fig.text(0.02, 0.9, '(a)', ha='center', fontsize=size2+10)
plt.savefig('Li_kpoint_convergence.png', dpi=400, bbox_inches='tight')

fig = plt.figure(figsize=(7,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
plt.scatter(PW_Li, E_PW_Li, c='b', label='Li @ Convergence check')
plt.plot(PW_Li, E_PW_Li,'b')
PW_Li_DFT=500
plt.plot([PW_Li_DFT,PW_Li_DFT],[np.min(E_PW_Li),np.max(E_PW_Li)],'k', label='Used value')
#plt.plot([3.4,3.4],[np.min(E_N_Li),np.max(E_N_Li)])
plt.xlabel(r'Plane Wave Cutoff [eV]',fontsize=size1)
plt.ylabel(r'$\Delta$E$_{~^*\!N}$ [eV]',fontsize=size1)
plt.legend(loc=4, fontsize=size2)
fig.text(0.02, 0.9, '(b)', ha='center', fontsize=size2+10)
plt.yticks([-1,-0.9,-0.8,-0.7],[r'$-1.0$',r'$-0.9$',r'$-0.8$',r'$-0.7$'])
plt.savefig('Li_PW_convergence.png', dpi=400, bbox_inches='tight')

#############################################################
#### N binding convergence.
##################################################################

L_Li=[3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
L_Ca=[5.2,5.3,5.4,5.5,5.6,5.7,5.8]
L_Mg=[2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7]

E_N_Li=[-1.4695,-1.3615,-1.2395,-0.7745,-0.7335,-0.6635,-0.6055,-0.5645,-0.5195,-0.5075]
E_N_Ca=[-1.9535,-1.9105,-1.8295,-1.7685,-1.7495,-1.643137,-1.53345]
E_N_Mg=[-0.755662,-0.6729,-0.6359,-0.5797,-0.50463,-0.47133,-0.4669,-0.443391,-0.465505]

fig = plt.figure(figsize=(7,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
plt.scatter(L_Li, E_N_Li, c='b', label='Li @ Convergence check')
plt.plot(L_Li, E_N_Li,'b')
L_Li_EXP=3.51
L_Li_DFT=3.49
plt.plot([L_Li_DFT,L_Li_DFT],[np.min(E_N_Li),np.max(E_N_Li)],'k', label='Optimized value')
#plt.plot([3.4,3.4],[np.min(E_N_Li),np.max(E_N_Li)])
plt.xlabel(r'Lattice constant [Ang]',fontsize=size1)
plt.ylabel(r'$\Delta$E$_{~^*\!N}$ [eV]',fontsize=size1)
plt.legend(loc=4, fontsize=size2)
fig.text(0.02, 0.9, '(a)', ha='center', fontsize=size2+10)
plt.savefig('Li_lattice_convergence.png', dpi=400, bbox_inches='tight')


fig = plt.figure(figsize=(7,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
plt.scatter(L_Ca, E_N_Ca,c='b', label='Ca @ Convergence check')
plt.plot(L_Ca, E_N_Ca,'b')
L_Ca_EXP=5.5884
L_Ca_DFT=5.58
plt.plot([L_Ca_DFT,L_Ca_DFT],[np.min(E_N_Ca),np.max(E_N_Ca)],'k', label='Optimized value')
#plt.plot([3.4,3.4],[np.min(E_N_Li),np.max(E_N_Li)])
plt.xlabel(r'Lattice constant [Ang]',fontsize=size1)
plt.ylabel(r'$\Delta$E$_{~^*\!N}$ [eV]',fontsize=size1)
plt.legend(loc=2, fontsize=size2)
fig.text(0.02, 0.9, '(b)', ha='center', fontsize=size2+10)
plt.savefig('Ca_lattice_convergence.png', dpi=400, bbox_inches='tight')


fig = plt.figure(figsize=(7,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
plt.scatter(L_Mg, E_N_Mg,c='b', label='Mg @ Convergence check')
plt.plot(L_Mg, E_N_Mg,'b')
L_Mg_EXP_a=3.2094
L_Mg_DFT_a=3.21
plt.plot([L_Mg_DFT_a,L_Mg_DFT_a],[np.min(E_N_Mg),np.max(E_N_Mg)],'k', label='Optimized value')
#plt.plot([3.4,3.4],[np.min(E_N_Li),np.max(E_N_Li)])
plt.xlabel(r'Lattice constant [Ang]',fontsize=size1)
plt.ylabel(r'$\Delta$E$_{~^*\!N}$ [eV]',fontsize=size1)
plt.legend(loc=4, fontsize=size2)
fig.text(0.02, 0.9, '(c)', ha='center', fontsize=size2+10)
plt.yticks([-0.7,-0.6,-0.5],[r'$-0.7$',r'$-0.6$',r'$-0.5$'])
plt.savefig('Mg_lattice_convergence.png', dpi=400, bbox_inches='tight')
