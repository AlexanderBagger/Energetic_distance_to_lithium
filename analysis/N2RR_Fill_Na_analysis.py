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



# Display where NAs exist
Where_is_NAs=np.where(df.isnull())[0]
print(Where_is_NAs)
print(df['Mname'][Where_is_NAs].unique())

#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################

from sklearn.linear_model import LinearRegression
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)

y_feature=['MxFy Formation Energy']
x_features=['Nitride Formation Energy','Oxide Formation Energy']


Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]

plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')

# adding number to df
#df[y_feature]=df[y_feature].fillna(pred[30][0])
print(df['Mname'][Where_is_NAs])


for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
y_feature=['Hydride Formation Energy']
x_features=['Nitride Formation Energy','Oxide Formation Energy','MxFy Formation Energy']


Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')



print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)




#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
y_feature=['MN3 Formation Energy']
x_features=['Nitride Formation Energy','Oxide Formation Energy','MxFy Formation Energy','Hydride Formation Energy']


Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')



print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
y_feature=['M3N Formation Energy']
x_features=['Nitride Formation Energy','Oxide Formation Energy','MxFy Formation Energy','Hydride Formation Energy', 'MN3 Formation Energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]

plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')


print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)

#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################

x_features.append(y_feature[0])
y_feature=['N_energy']


Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), 'N Binding Energy' +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')



print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
x_features.append(y_feature[0])
y_feature=['N2_energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), 'N2 Binding Energy' +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')



print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)



#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
x_features.append(y_feature[0])
y_feature=['NH2_energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), 'NH2 Binding Energy' +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')



print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
x_features.append(y_feature[0])
y_feature=['NH3_energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]


plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), 'NH3 Binding Energy' +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')


print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)


#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
x_features.append(y_feature[0])
y_feature=['MxOyHz Formation Energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]

plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')


print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)

#############################################################
#### Calculating distantance to Li - Formation energies
##################################################################
x_features.append(y_feature[0])
y_feature=['MxCyOz Formation Energy']

Where_is_NAs=np.where(df[y_feature].isnull())[0]
df_small=df.dropna(subset = y_feature)
regr = LinearRegression()
regr.fit(df_small[x_features], df_small[y_feature])
predictions=regr.predict(df[x_features])

pred = predictions.tolist()

pred_plot=pred.copy()
for k in np.flipud(Where_is_NAs):
    del pred_plot[k]

plt.figure()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size2-4); plt.yticks(fontsize = size2-4)
plt.text(df_small[y_feature].values.min(),df_small[y_feature].values.max()-0.2*(df_small[y_feature].values.max()-df_small[y_feature].values.min()), y_feature[0] +'\n' + r'R$^2$=' + str(np.round(regr.score(df_small[x_features], df_small[y_feature]),2)),fontsize=size2-4)
plt.scatter(df_small[y_feature].values, pred_plot)
for k in np.flipud(Where_is_NAs):
    plt.scatter(pred[k], pred[k],c='r')
    plt.text(pred[k][0], pred[k][0],df['Mname'][k],fontsize=size2-6)
plt.plot([df_small[y_feature].values.min(), df_small[y_feature].values.max()],[df_small[y_feature].values.min(),df_small[y_feature].values.max()])
plt.xlabel(r'DFT value [eV]',fontsize=size2)
plt.ylabel(r'Predicted value [eV]',fontsize=size2)
plt.savefig('prediction_fill_'+y_feature[0]+'.png', dpi=400, bbox_inches='tight')


print(Where_is_NAs)
print(df['Mname'][Where_is_NAs])

for k in Where_is_NAs: 
    df.at[k, y_feature[0]] = pred[k][0]

# looking at the updated df
list_of_NAs=df[Features_small].isna().sum()
print(list_of_NAs)



#############################################################
#### PCA analysis - formation energies - with SHE
##################################################################
Features_small.append('Mname')
Features_small.append('Dissociation')
Features_small.append('V SHE')

df_small=df[Features_small]
df_small=df_small.dropna()

# Separating out the features
features=Features_small[:-3]
features_label=Features_small_label

# Adding SHE
features.append('V SHE')
features_label.append(r'V$_{SHE}$')

x = df_small.loc[:, features].values
# Standardizing the features
test = StandardScaler().fit_transform(x)
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(test)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
finalDf = principalDf
x=finalDf.loc[:,['principal component 1']].values
y=finalDf.loc[:,['principal component 2']].values
text=df_small.loc[:,['Mname']].values
scalex = 1.0/(x.max() - x.min())
scaley = 1.0/(y.max() - y.min())

fig=plt.figure(figsize = (8,8))
plt.locator_params(axis='x',nbins=5);plt.grid(True);plt.locator_params(axis='y',nbins=5)    
for k in range(0,len(x)):
    if text[k,0]=='Li':
        plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot+300,edgecolors= "black",alpha=1.0,zorder=1)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2,zorder=2)
    elif df_small['Dissociation'].values[k]==True:
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]==False:
        p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    else:
        plt.scatter(x[k]*scalex, y[k]*scaley, color='k',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)


coeff=np.transpose(pca.components_)
n = coeff.shape[0]
for i in range(n):
    if features[i][:7]=='Hydride':
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='Nitride Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.5, coeff[i,1]*1.5,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.55, coeff[i,1] * 1.55, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='V SHE' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'k',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'k', ha = 'center', va = 'center',fontsize = s3)

    else : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

#plt.text(-0.6,0.6,'=%s' % (pca.explained_variance_ratio_.sum()*100).round() +'%', fontsize=20)
plt.xlim([-0.75,0.75])
plt.ylim([-0.75,0.75])
plt.legend([p1,p2],['N-N cleaved phase','N-N coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=3, fontsize=size1-8,markerscale=1.0)
plt.xticks(fontsize=size2)
plt.yticks(fontsize=size2)
plt.xlabel('Principal Component 1 (' +str(np.round(pca.explained_variance_ratio_[0]*100,0))[:2]+' %)',fontsize=size1)
plt.ylabel('Principal Component 2 (' +str(np.round(pca.explained_variance_ratio_[1]*100,0))[:2]+' %)',fontsize=size1)
fig.text(0.02, 0.9, '(d)', ha='center', fontsize=size2+10)
#plt.savefig('PCA_features_small_LinearReg_Vshe.png', dpi=400, bbox_inches='tight')
#plt.savefig('PC_analysis0.png', dpi=900, bbox_inches='tight')


fig=plt.figure(figsize = (8,8))
plt.locator_params(axis='x',nbins=5);plt.grid(True);plt.locator_params(axis='y',nbins=5)    
for k in range(0,len(x)):
    if text[k,0]=='Li' or text[k,0]=='Ca':
        plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot+300,edgecolors= "black",alpha=1.0,zorder=1)
        plt.text(x[k]*scalex+0.04, y[k]*scaley-0.02, text[k,0], fontsize=size2,zorder=2)

    elif text[k,0]=='Hg' or text[k,0]=='Pt' or text[k,0]=='Y' or text[k,0]=='In' or text[k,0]=='Zr' or text[k,0]=='Ru' or text[k,0]=='Ir' or text[k,0]=='Au' or text[k,0]=='Ta' or text[k,0]=='Fe' or text[k,0]=='Nb' or text[k,0]=='Rh' or text[k,0]=='Hf' or text[k,0]=='Re' or text[k,0]=='Ga' or text[k,0]=='Pd' or text[k,0]=='Be':                
        if text[k,0]=='Rb' or text[k,0]=='Tl':
            p3=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
        elif text[k,0]=='Co':
            p4=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
        elif text[k,0]=='Sc' or text[k,0]=='Cd' or text[k,0]=='Ag':
            p3=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley-0.05, text[k,0], fontsize=size2)
        elif df_small['Dissociation'].values[k]==True:
            p4=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
        elif df_small['Dissociation'].values[k]==False:
            p3=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
        else:
            plt.scatter(x[k]*scalex, y[k]*scaley, color='k',s=sdot,marker='*',edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)

    else:
        if text[k,0]=='Rb' or text[k,0]=='Tl':
            p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
        elif text[k,0]=='Co':
            p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
        elif text[k,0]=='Sc' or text[k,0]=='Cd' or text[k,0]=='Ag':
            p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley-0.05, text[k,0], fontsize=size2)
        elif df_small['Dissociation'].values[k]==True:
            p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
        elif df_small['Dissociation'].values[k]==False:
            p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
        else:
            plt.scatter(x[k]*scalex, y[k]*scaley, color='k',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
            plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)


coeff=np.transpose(pca.components_)
n = coeff.shape[0]
for i in range(n):
    if features[i][:7]=='Hydride':
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.6, coeff[i,1]*1.6,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.65, coeff[i,1] * 1.65, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='Nitride Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.7, coeff[i,1]*1.7,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.75, coeff[i,1] * 1.75, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='Oxide Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.7, coeff[i,1]*1.7,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.75, coeff[i,1] * 1.75, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='N2_energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.6, coeff[i,1]*1.6,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.65, coeff[i,1] * 1.65, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)


    elif features[i]=='NH2_energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.7, coeff[i,1]*1.7,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.75, coeff[i,1] * 1.75, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='NH3_energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.3, coeff[i,1]*1.3,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.35, coeff[i,1] * 1.35, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)


    elif features[i]=='M3N Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.8, coeff[i,1]*1.8,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.85, coeff[i,1] * 1.85+0.025, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='MxOyHz Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.7, coeff[i,1]*1.7,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.75, coeff[i,1] * 1.75-0.015, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)



    elif features[i]=='V SHE' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'k',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'k', ha = 'center', va = 'center',fontsize = s3)

    elif features[i]=='N_energy' or features[i]=='MN3 Formation Energy':
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

    else : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.7, coeff[i,1]*1.7,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.75, coeff[i,1] * 1.75, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

plt.xlim([-0.7,0.8])
plt.ylim([-0.8,0.7])
plt.legend([p1,p2,p4,p3],['Cleaved phase','Coupling phase','W/ predictions','W/ predictions'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0, ncol=2, columnspacing=0.6, handletextpad=0.05)
plt.xticks(fontsize=size2)
plt.yticks(fontsize=size2)
plt.xlabel('Principal Component 1 (' +str(np.round(pca.explained_variance_ratio_[0]*100,0))[:2]+' %)',fontsize=size1)
plt.ylabel('Principal Component 2 (' +str(np.round(pca.explained_variance_ratio_[1]*100,0))[:2]+' %)',fontsize=size1)
fig.text(0.02, 0.9, '(b)', ha='center', fontsize=size2+10)
plt.savefig('PCA_features_small_Vshe_MS2.png', dpi=900, bbox_inches='tight')
#plt.savefig('PC_analysis0.png', dpi=900, bbox_inches='tight')

Features.append('Mname')
Features.append('Dissociation')
Features.append('V SHE')
F1=[ 'Mname',
 'Dissociation',
 'V SHE',
 'Nitride Formation Energy',
 'Hydride Formation Energy',
 'Oxide Formation Energy',
 'MxOyHz Formation Energy',
 'MxCyOz Formation Energy',
 'MxFy Formation Energy',]
df[F1].round(2).to_csv('dataset_F1.csv',index=False)

F2=['Mname',
 'Dissociation',
 'V SHE',
 'MN3 Formation Energy',
 'M3N Formation Energy',
 'N_energy',
 'N2_energy',
 'NH2_energy',
 'NH3_energy']
df[F2].round(2).to_csv('dataset_F2.csv',index=False)

F3=['Mname',
 'Dissociation',
 'V SHE',
 'MxOyHz Reaction',
 'MxOy Reaction',
 'MxHy Reaction',
 'Mx Reaction']
df[F3].round(2).to_csv('dataset_F3.csv',index=False)



