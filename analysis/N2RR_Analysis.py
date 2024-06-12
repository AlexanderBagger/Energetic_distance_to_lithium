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
#### load in molecule data 
##################################################################
Mol=ase.db.connect('../databases/molecules.db') # PW800
ENH3=Mol.get(name='NH3').energy
EH2=Mol.get(name='H2').energy
EN2=Mol.get(name='N2').energy
EH2O=Mol.get(name='H2O').energy
ECO=Mol.get(name='CO').energy
EHF=Mol.get(name='HF').energy



Mol_PW500=ase.db.connect('../databases/molecules_PW500.db')
ENH3_PW500=Mol_PW500.get(name='NH3').energy
EH2_PW500=Mol_PW500.get(name='H2').energy
EN2_PW500=Mol_PW500.get(name='N2').energy


dict_bulk={}
dM=ase.db.connect('../databases/bulk.db') # PW800

# Here I generate some lists to save data
bulk_Mname=[]
bulk_Mname_count=[]
bulk_energy=[]
bulk_name=[]

bulk2_Mname=[]
bulk2_Mname_count=[]
bulk2_atom2=[]
bulk2_atom2_count=[]
bulk2_energy=[]
bulk2_name=[]

MxOyHz_name=[]
MxOyHz_Mname=[]
MxOyHz_count=[]
MxOyHz_energy=[]
MxOyHz_M=[]
MxOyHz_O=[]
MxOyHz_H=[]

MxHy_name=[]
MxHy_Mname=[]
MxHy_count=[]
MxHy_energy=[]
MxHy_M=[]
MxHy_H=[]

MxOy_name=[]
MxOy_Mname=[]
MxOy_count=[]
MxOy_energy=[]
MxOy_M=[]
MxOy_O=[]

MxNy_name=[]
MxNy_Mname=[]
MxNy_count=[]
MxNy_energy=[]
MxNy_M=[]
MxNy_N=[]

Mx_name=[]
Mx_Mname=[]
Mx_count=[]
Mx_energy=[]
Mx_M=[]

#############################################################
#### load in structure data 
##################################################################

for row in dM.select(relax='unitcell'):
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/bulk.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    
# If M_x O_y H_z
    if (any('H' in string for string in data)
        and any('O' in string for string in data) and sym.count('H')>0 and sym.count('O')>0
        and name!='Hf'):
        #print('MxOyHz' ,name)
        MxOyHz_name.append(name)
        if sym[0]=='H':
            if name[:1]=='W' or name[:1]=='B' or name[:1]=='Y' or name[:1]=='K' or name[:1]=='V':
                MxOyHz_Mname.append(name[:1])
            else:
                MxOyHz_Mname.append(name[:2])
        else:
            MxOyHz_Mname.append(sym[0])
        MxOyHz_count.append(row.natoms)
        MxOyHz_energy.append(row.energy)
        MxOyHz_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxOyHz_O.append(sym.count('O'))
        MxOyHz_H.append(sym.count('H'))

    
# If M_x H_z
    elif any('H' in string for string in data) and sym.count('O')==0 and sym.count('H')>0:
        #print('MxHy' ,name)
        MxHy_name.append(name)
        if sym[0]=='H':
            test1=sym[-1]
            MxHy_Mname.append(sym[-1])
        else:
            test1=sym[0]
            MxHy_Mname.append(sym[0])
        MxHy_count.append(row.natoms)
        MxHy_energy.append(row.energy)
        MxHy_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxHy_H.append(sym.count('H'))
        
        #print(name)
        #print(test1)
# If M_x O_y
    elif sym.count('H')==0 and any('O' in string for string in data)==True and sym.count('O')>0:
        #print('MxOy' ,name)
        MxOy_name.append(name)
        MxOy_Mname.append(sym[0])
        MxOy_count.append(row.natoms)
        MxOy_energy.append(row.energy)
        MxOy_M.append(row.natoms-sym.count('O')-sym.count('H'))
        MxOy_O.append(sym.count('O'))

        

# If M_x N_y
    elif any('N' in string for string in data) and sym.count('N')>0 and sym.count('H')==0:
        #print('MxNy' ,name)
        MxNy_name.append(name)
        MxNy_Mname.append(sym[0])
        MxNy_count.append(row.natoms)
        MxNy_energy.append(row.energy)
        MxNy_M.append(row.natoms-sym.count('N')-sym.count('H'))
        MxNy_N.append(sym.count('N'))

# If M_x
    elif sym.count('H')==0 and sym.count('O')==0 and sym.count('N')==0:
        #print('Mx' ,name)
        Mx_name.append(name)
        Mx_Mname.append(sym[0])
        Mx_count.append(row.natoms)
        Mx_energy.append(row.energy)
        Mx_M.append(row.natoms-sym.count('O')-sym.count('H'))

        
    else:
        print('REST' ,name)



#############################################################
#### load in MxFy & MxCyOx structure data 
##################################################################
dMF=ase.db.connect('../databases/MxFy.db') # PW800
dMCO3=ase.db.connect('../databases/MxCyOz.db') # PW800

MxFy_name=[]
MxFy_Mname=[]
MxFy_count=[]
MxFy_energy=[]
MxFy_M=[]
MxFy_F=[]

for row in dMF.select(relax='unitcell'):
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/MxFy.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()

    MxFy_name.append(name)
    MxFy_Mname.append(sym[0])
    MxFy_count.append(row.natoms)
    MxFy_energy.append(row.energy)
    MxFy_M.append(row.natoms-sym.count('F'))
    MxFy_F.append(sym.count('F'))

    #print(name)
    
MxCyOz_name=[]
MxCyOz_Mname=[]
MxCyOz_count=[]
MxCyOz_energy=[]
MxCyOz_M=[]
MxCyOz_C=[]
MxCyOz_O=[]

for row in dMCO3.select(relax='unitcell'):
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/MxCyOz.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    
    MxCyOz_name.append(name)
    MxCyOz_Mname.append(sym[0])
    MxCyOz_count.append(row.natoms)
    MxCyOz_energy.append(row.energy)
    MxCyOz_M.append(row.natoms-sym.count('C')-sym.count('O'))
    MxCyOz_C.append(sym.count('C'))
    MxCyOz_O.append(sym.count('O'))


#############################################################
#### Create pandas data frames
##################################################################

d_MxOyHz = {
'MxOyHz_name': MxOyHz_name,
'Mname':MxOyHz_Mname,
'MxOyHz_count':MxOyHz_count,
'MxOyHz_energy':MxOyHz_energy,
'MxOyHz_M':MxOyHz_M,
'MxOyHz_O':MxOyHz_O,
'MxOyHz_H':MxOyHz_H}
df_MxOyHz = pd.DataFrame(data=d_MxOyHz)

d_MxHy = {
'MxHy_name': MxHy_name,
'Mname':MxHy_Mname,
'MxHy_count':MxHy_count,
'MxHy_energy':MxHy_energy,
'MxHy_M':MxHy_M,
'MxHy_H':MxHy_H}
df_MxHy = pd.DataFrame(data=d_MxHy)

d_MxOy = {
'MxOy_name': MxOy_name,
'Mname':MxOy_Mname,
'MxOy_count':MxOy_count,
'MxOy_energy':MxOy_energy,
'MxOy_M':MxOy_M,
'MxOy_O':MxOy_O}
df_MxOy = pd.DataFrame(data=d_MxOy)

d_MxNy = {
'MxNy_name': MxNy_name,
'Mname':MxNy_Mname,
'MxNy_count':MxNy_count,
'MxNy_energy':MxNy_energy,
'MxNy_M':MxNy_M,
'MxNy_N':MxNy_N}
df_MxNy = pd.DataFrame(data=d_MxNy)

d_Mx = {
'Mx_name': Mx_name,
'Mname':Mx_Mname,
'Mx_count':Mx_count,
'Mx_energy':Mx_energy,
'Mx_M':Mx_M}
df_Mx = pd.DataFrame(data=d_Mx)

d_MxCyOz = {
'MxCyOz_name': MxCyOz_name,
'Mname':MxCyOz_Mname,
'MxCyOz_count':MxCyOz_count,
'MxCyOz_energy':MxCyOz_energy,
'MxCyOz_M':MxCyOz_M,
'MxCyOz_C':MxCyOz_C,
'MxCyOz_O':MxCyOz_O}
df_MxCyOz = pd.DataFrame(data=d_MxCyOz)

d_MxFy = {
'MxFy_name': MxFy_name,
'Mname':MxFy_Mname,
'MxFy_count':MxFy_count,
'MxFy_energy':MxFy_energy,
'MxFy_M':MxFy_M,
'MxFy_F':MxFy_F}
df_MxFy = pd.DataFrame(data=d_MxFy)

# Creating big datadframe (merge the pandas frames)
df=pd.merge(df_Mx, df_MxNy, on="Mname", how="left")
df=pd.merge(df, df_MxOy, on="Mname", how="left")
df=pd.merge(df, df_MxHy, on="Mname", how="left")
df=pd.merge(df, df_MxOyHz, on="Mname", how="left")
df=pd.merge(df, df_MxCyOz, on="Mname", how="left")
df=pd.merge(df, df_MxFy, on="Mname", how="left")



#############################################################
#### Calculations of formation energies and reaction energies
##################################################################

# Calculate formation energies
df['Nitride Formation Energy']=(df['MxNy_energy']-df['MxNy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxNy_N']*EN2*0.5)/df['MxNy_count']

df['Hydride Formation Energy']=(df['MxHy_energy']-df['MxHy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxHy_H']*EH2*0.5)/df['MxHy_count']

df['Oxide Formation Energy']=(df['MxOy_energy']-df['MxOy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxOy_O']*(EH2O-EH2))/df['MxOy_count']

df['MxOyHz Formation Energy']=(df['MxOyHz_energy']-df['MxOyHz_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxOyHz_O']*(EH2O-EH2)-df['MxOyHz_H']*EH2*0.5)/df['MxOyHz_count']

df['MxCyOz Formation Energy']=(df['MxCyOz_energy']-df['MxCyOz_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxCyOz_O']*(EH2O-EH2)-df['MxCyOz_C']*(ECO-(EH2O-EH2)))/df['MxCyOz_count']

df['MxFy Formation Energy']=(df['MxFy_energy']-df['MxFy_M']*df['Mx_energy']/df['Mx_count'] 
    -df['MxFy_F']*(EHF-0.5*EH2))/df['MxFy_count']

# calculating reaction energies:
df['MxOyHz Reaction']=(df['MxOyHz_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']-df['MxOyHz_O']*EH2O
    -(df['MxOyHz_H']+3*df['MxNy_N']-2*df['MxOyHz_O'])*EH2*0.5
    -(df['MxOyHz_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['MxOy Reaction']=(df['MxOy_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']-df['MxOy_O']*EH2O
    -(3*df['MxNy_N']-2*df['MxOy_O'])*EH2*0.5
    -(df['MxOy_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['MxHy Reaction']=(df['MxHy_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']
    -(3*df['MxNy_N']+df['MxHy_H'])*EH2*0.5
    -(df['MxHy_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

df['Mx Reaction']=(df['Mx_energy']+df['MxNy_N']*ENH3 
    -df['MxNy_energy']
    -(3*df['MxNy_N'])*EH2*0.5
    -(df['Mx_M']-df['MxNy_M'])*(df['Mx_energy']/df['Mx_M']))/(df['MxNy_N'])

###############################
# Load X3N & XN3 energies
###############################
X3N=ase.db.connect('../databases/X3N_bulk.db')  # PW800 
XN3=ase.db.connect('../databases/XN3_bulk.db')  # PW800

#atoms=read('../databases/XN3_bulk.db@')
#view(atoms)
X3N_name=[]
X3N_energy=[]
X3N_M=[]
X3N_N=[]
X3N_mass=[]
XN3_name=[]
XN3_energy=[]
XN3_M=[]
XN3_N=[]
XN3_mass=[]

for row in X3N.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/X3N_bulk.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    X3N_name.append(sym[0])
    Esub=3*df_Mx['Mx_energy'].loc[df_Mx['Mname'] == sym[0]].values[0]/df_Mx['Mx_M'].loc[df_Mx['Mname'] == sym[0]].values[0]
    X3N_energy.append(row.energy-Esub-0.5*EN2)
    X3N_M.append(row.natoms-sym.count('N'))
    X3N_N.append(sym.count('N'))
    X3N_mass.append(row.mass)
    
for row in XN3.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/XN3_bulk.db@id=%s' %row.id)[0]
    sym=Atoms.get_chemical_symbols()
    XN3_name.append(sym[0])
    Esub=3*df_Mx['Mx_energy'].loc[df_Mx['Mname'] == sym[0]].values[0]/df_Mx['Mx_M'].loc[df_Mx['Mname'] == sym[0]].values[0]
    XN3_energy.append(row.energy-4.5*EN2-Esub)
    XN3_M.append(row.natoms-sym.count('N'))
    XN3_N.append(sym.count('N'))
    XN3_mass.append(row.mass)

d_XN3 = {
'Mname':XN3_name,
'MN3 Formation Energy':XN3_energy}
df_XN3 = pd.DataFrame(data=d_XN3)

d_X3N = {
'Mname':X3N_name,
'M3N Formation Energy':X3N_energy}
df_X3N = pd.DataFrame(data=d_X3N)

df=pd.merge(df, df_XN3, on="Mname", how="left")
df=pd.merge(df, df_X3N, on="Mname", how="left")



###############################
# Load binding energies PW500
###############################
dict_bulk={}

bcc_slab=ase.db.connect('../databases/bcc_slab.db')
fcc_slab=ase.db.connect('../databases/fcc_slab.db')
hcp_slab=ase.db.connect('../databases/hcp_slab.db')

bcc_slab_N=ase.db.connect('../databases/bcc_slab_N.db')
fcc_slab_N=ase.db.connect('../databases/fcc_slab_N.db')
hcp_slab_N=ase.db.connect('../databases/hcp_slab_N.db')

bcc_slab_N2=ase.db.connect('../databases/bcc_slab_N2.db')
fcc_slab_N2=ase.db.connect('../databases/fcc_slab_N2.db')
hcp_slab_N2=ase.db.connect('../databases/hcp_slab_N2.db')

bcc_slab_NH2=ase.db.connect('../databases/bcc_slab_NH2.db')
fcc_slab_NH2=ase.db.connect('../databases/fcc_slab_NH2.db')
hcp_slab_NH2=ase.db.connect('../databases/hcp_slab_NH2.db')

bcc_slab_NH3=ase.db.connect('../databases/bcc_slab_NH3.db')
fcc_slab_NH3=ase.db.connect('../databases/fcc_slab_NH3.db')
hcp_slab_NH3=ase.db.connect('../databases/hcp_slab_NH3.db')


bcc_name=[]
bcc_energy=[]
bcc_energy_N=[]
bcc_energy_N2=[]
bcc_energy_NH2=[]
bcc_energy_NH3=[]

for row in bcc_slab.select():
    name=row.formula
    if name=='Mn20':
        print('Not Mn')
    else: 
        data=seperate_string_number(name)
        Atoms=read('../databases/bcc_slab.db@formula=%s' %name)[0]
        bcc_name.append(Atoms.get_chemical_symbols()[0])
        bcc_energy.append(Atoms.get_potential_energy())
        bcc_energy_N.append(read('../databases/bcc_slab_N.db@formula=%s' %name+'N')[0].get_potential_energy())
        bcc_energy_N2.append(read('../databases/bcc_slab_N2.db@formula=%s' %name+'N2')[0].get_potential_energy())
        bcc_energy_NH2.append(read('../databases/bcc_slab_NH2.db@formula=%s' %name+'NH2')[0].get_potential_energy())    
        bcc_energy_NH3.append(read('../databases/bcc_slab_NH3.db@formula=%s' %name+'NH3')[0].get_potential_energy())
            
fcc_name=[]
fcc_energy=[]
fcc_energy_N=[]
fcc_energy_N2=[]
fcc_energy_NH2=[]
fcc_energy_NH3=[]

for row in fcc_slab.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/fcc_slab.db@formula=%s' %name)[0]
    fcc_name.append(Atoms.get_chemical_symbols()[0])
    fcc_energy.append(Atoms.get_potential_energy())
    fcc_energy_N.append(read('../databases/fcc_slab_N.db@formula=%s' %name+'N')[0].get_potential_energy())
    fcc_energy_N2.append(read('../databases/fcc_slab_N2.db@formula=%s' %name+'N2')[0].get_potential_energy())
    fcc_energy_NH2.append(read('../databases/fcc_slab_NH2.db@formula=%s' %name+'NH2')[0].get_potential_energy())    
    fcc_energy_NH3.append(read('../databases/fcc_slab_NH3.db@formula=%s' %name+'NH3')[0].get_potential_energy())

hcp_name=[]
hcp_energy=[]
hcp_energy_N=[]
hcp_energy_N2=[]
hcp_energy_NH2=[]
hcp_energy_NH3=[]

for row in hcp_slab.select():
    name=row.formula
    data=seperate_string_number(name)
    Atoms=read('../databases/hcp_slab.db@formula=%s' %name)[0]
    hcp_name.append(Atoms.get_chemical_symbols()[0])
    hcp_energy.append(Atoms.get_potential_energy())
    hcp_energy_N.append(read('../databases/hcp_slab_N.db@formula=%s' %name+'N')[0].get_potential_energy())
    hcp_energy_N2.append(read('../databases/hcp_slab_N2.db@formula=%s' %name+'N2')[0].get_potential_energy())
    hcp_energy_NH2.append(read('../databases/hcp_slab_NH2.db@formula=%s' %name+'NH2')[0].get_potential_energy())    
    hcp_energy_NH3.append(read('../databases/hcp_slab_NH3.db@formula=%s' %name+'NH3')[0].get_potential_energy())



#############################################################
#### Make N pandas frames, and combine it with the rest
##################################################################

d_bcc_N = {
'Mname': bcc_name,
'N_energy':np.asarray(bcc_energy_N)-np.asarray(bcc_energy)-0.5*EN2_PW500}
df_bcc_N = pd.DataFrame(data=d_bcc_N)

d_fcc_N = {
'Mname': fcc_name,
'N_energy':np.asarray(fcc_energy_N)-np.asarray(fcc_energy)-0.5*EN2_PW500}
df_fcc_N = pd.DataFrame(data=d_fcc_N)

d_hcp_N = {
'Mname': hcp_name,
'N_energy':np.asarray(hcp_energy_N)-np.asarray(hcp_energy)-0.5*EN2_PW500}
df_hcp_N = pd.DataFrame(data=d_hcp_N)

df_N = df_bcc_N.append(df_fcc_N, ignore_index=True)
df_N = df_N.append(df_hcp_N, ignore_index=True)
df=pd.merge(df, df_N, on="Mname", how="left")


d_bcc_N2 = {
'Mname': bcc_name,
'N2_energy':np.asarray(bcc_energy_N2)-np.asarray(bcc_energy)-EN2_PW500}
df_bcc_N2 = pd.DataFrame(data=d_bcc_N2)

d_fcc_N2 = {
'Mname': fcc_name,
'N2_energy':np.asarray(fcc_energy_N2)-np.asarray(fcc_energy)-EN2_PW500}
df_fcc_N2 = pd.DataFrame(data=d_fcc_N2)

d_hcp_N2 = {
'Mname': hcp_name,
'N2_energy':np.asarray(hcp_energy_N2)-np.asarray(hcp_energy)-EN2_PW500}
df_hcp_N2 = pd.DataFrame(data=d_hcp_N2)

df_N2 = df_bcc_N2.append(df_fcc_N2, ignore_index=True)
df_N2 = df_N2.append(df_hcp_N2, ignore_index=True)
df=pd.merge(df, df_N2, on="Mname", how="left")


d_bcc_NH2 = {
'Mname': bcc_name,
'NH2_energy':np.asarray(bcc_energy_NH2)-np.asarray(bcc_energy)-0.5*EN2_PW500-EH2_PW500}
df_bcc_NH2 = pd.DataFrame(data=d_bcc_NH2)

d_fcc_NH2 = {
'Mname': fcc_name,
'NH2_energy':np.asarray(fcc_energy_NH2)-np.asarray(fcc_energy)-0.5*EN2_PW500-EH2_PW500}
df_fcc_NH2 = pd.DataFrame(data=d_fcc_NH2)

d_hcp_NH2 = {
'Mname': hcp_name,
'NH2_energy':np.asarray(hcp_energy_NH2)-np.asarray(hcp_energy)-0.5*EN2_PW500-EH2_PW500}
df_hcp_NH2 = pd.DataFrame(data=d_hcp_NH2)

df_NH2 = df_bcc_NH2.append(df_fcc_NH2, ignore_index=True)
df_NH2 = df_NH2.append(df_hcp_NH2, ignore_index=True)
df=pd.merge(df, df_NH2, on="Mname", how="left")

d_bcc_NH3 = {
'Mname': bcc_name,
'NH3_energy':np.asarray(bcc_energy_NH3)-np.asarray(bcc_energy)-0.5*EN2_PW500-1.5*EH2_PW500}
df_bcc_NH3 = pd.DataFrame(data=d_bcc_NH3)

fcc_energy_small=fcc_energy
d_fcc_NH3 = {
'Mname': fcc_name,
'NH3_energy':np.asarray(fcc_energy_NH3)-np.asarray(fcc_energy_small)-0.5*EN2_PW500-1.5*EH2_PW500}
df_fcc_NH3 = pd.DataFrame(data=d_fcc_NH3)


d_hcp_NH3 = {
'Mname': hcp_name,
'NH3_energy':np.asarray(hcp_energy_NH3)-np.asarray(hcp_energy)-0.5*EN2_PW500-1.5*EH2_PW500}
df_hcp_NH3 = pd.DataFrame(data=d_hcp_NH3)

df_NH3 = df_bcc_NH3.append(df_fcc_NH3, ignore_index=True)
df_NH3 = df_NH3.append(df_hcp_NH3, ignore_index=True)
df=pd.merge(df, df_NH3, on="Mname", how="left")

#############################################################
#### Here we add a V SHE collumn and add the SHE potential for the elements
#### Here we add a dissocciation collumn for the phases and write this data
##################################################################

# initiate V SHE and Dissociation
df['V SHE']=-2.0


# # from standard electro potential (data_page)
df['V SHE'].loc[df['Mname']=='Rb']=-2.98
df['V SHE'].loc[df['Mname']=='Mg']=-2.372
df['V SHE'].loc[df['Mname']=='Al']=-1.662
df['V SHE'].loc[df['Mname']=='K']=-2.931
df['V SHE'].loc[df['Mname']=='Ca']=-2.868
df['V SHE'].loc[df['Mname']=='Sr']=-2.899
df['V SHE'].loc[df['Mname']=='B']=-1.79 # From ion state
df['V SHE'].loc[df['Mname']=='Be']=-1.847
df['V SHE'].loc[df['Mname']=='Na']=-2.71
df['V SHE'].loc[df['Mname']=='Li']=-3.04
df['V SHE'].loc[df['Mname']=='Cs']=-3.026
df['V SHE'].loc[df['Mname']=='Ba']=-2.912
df['V SHE'].loc[df['Mname']=='Ga']=-0.53
df['V SHE'].loc[df['Mname']=='In']=-0.34
df['V SHE'].loc[df['Mname']=='Tl']=-0.34
df['V SHE'].loc[df['Mname']=='Sc']=-2.077
df['V SHE'].loc[df['Mname']=='Y']=-2.372
df['V SHE'].loc[df['Mname']=='Ti']=-1.63
df['V SHE'].loc[df['Mname']=='Zr']=-1.45
df['V SHE'].loc[df['Mname']=='Hf']=-1.724
df['V SHE'].loc[df['Mname']=='V']=-1.13
df['V SHE'].loc[df['Mname']=='Nb']=-1.099
df['V SHE'].loc[df['Mname']=='Ta']=-0.6
df['V SHE'].loc[df['Mname']=='Cr']=-0.74
df['V SHE'].loc[df['Mname']=='Mo']=-0.15
df['V SHE'].loc[df['Mname']=='W']=-0.12
df['V SHE'].loc[df['Mname']=='Mn']=-1.185
df['V SHE'].loc[df['Mname']=='Re']=0.3
df['V SHE'].loc[df['Mname']=='Fe']=-0.44
df['V SHE'].loc[df['Mname']=='Ru']=0.455
df['V SHE'].loc[df['Mname']=='Os']=0.65
df['V SHE'].loc[df['Mname']=='Co']=-0.28
df['V SHE'].loc[df['Mname']=='Rh']=0.76
df['V SHE'].loc[df['Mname']=='Ir']=1.0
df['V SHE'].loc[df['Mname']=='Ni']=-0.25
df['V SHE'].loc[df['Mname']=='Pd']=0.915
df['V SHE'].loc[df['Mname']=='Pt']=1.188
df['V SHE'].loc[df['Mname']=='Cu']=0.337
df['V SHE'].loc[df['Mname']=='Ag']=0.7996
df['V SHE'].loc[df['Mname']=='Au']=1.52
df['V SHE'].loc[df['Mname']=='Zn']=-0.7618
df['V SHE'].loc[df['Mname']=='Cd']=-0.4
df['V SHE'].loc[df['Mname']=='Hg']=0.85


# # # from standard electro potential (data_page)
# df['HSAB']=0.0
# df['HSAB'].loc[df['Mname']=='Rb']=np.nan
# df['HSAB'].loc[df['Mname']=='Mg']=2.42
# df['HSAB'].loc[df['Mname']=='Al']=6.01
# df['HSAB'].loc[df['Mname']=='K']=np.nan
# df['HSAB'].loc[df['Mname']=='Ca']=2.33
# df['HSAB'].loc[df['Mname']=='Sr']=2.21
# df['HSAB'].loc[df['Mname']=='B']=np.nan
# df['HSAB'].loc[df['Mname']=='Be']=3.75
# df['HSAB'].loc[df['Mname']=='Na']=0.0
# df['HSAB'].loc[df['Mname']=='Li']=0.49
# df['HSAB'].loc[df['Mname']=='Cs']=np.nan
# df['HSAB'].loc[df['Mname']=='Ba']=1.89
# df['HSAB'].loc[df['Mname']=='Ga']=1.45
# df['HSAB'].loc[df['Mname']=='In']=np.nan
# df['HSAB'].loc[df['Mname']=='Tl']=-1.88
# df['HSAB'].loc[df['Mname']=='Sc']=np.nan
# df['HSAB'].loc[df['Mname']=='Y']=np.nan
# df['HSAB'].loc[df['Mname']=='Ti']=4.35
# df['HSAB'].loc[df['Mname']=='Zr']=np.nan
# df['HSAB'].loc[df['Mname']=='Hf']=np.nan
# df['HSAB'].loc[df['Mname']=='V']=np.nan
# df['HSAB'].loc[df['Mname']=='Nb']=np.nan
# df['HSAB'].loc[df['Mname']=='Ta']=np.nan
# df['HSAB'].loc[df['Mname']=='Cr']=2.06
# df['HSAB'].loc[df['Mname']=='Mo']=np.nan
# df['HSAB'].loc[df['Mname']=='W']=np.nan
# df['HSAB'].loc[df['Mname']=='Mn']=np.nan
# df['HSAB'].loc[df['Mname']=='Re']=np.nan
# df['HSAB'].loc[df['Mname']=='Fe']=2.22
# df['HSAB'].loc[df['Mname']=='Ru']=np.nan
# df['HSAB'].loc[df['Mname']=='Os']=np.nan
# df['HSAB'].loc[df['Mname']=='Co']=np.nan
# df['HSAB'].loc[df['Mname']=='Rh']=np.nan
# df['HSAB'].loc[df['Mname']=='Ir']=np.nan
# df['HSAB'].loc[df['Mname']=='Ni']=0.29
# df['HSAB'].loc[df['Mname']=='Pd']=np.nan
# df['HSAB'].loc[df['Mname']=='Pt']=np.nan
# df['HSAB'].loc[df['Mname']=='Cu']=-0.55
# df['HSAB'].loc[df['Mname']=='Ag']=-2.82
# df['HSAB'].loc[df['Mname']=='Au']=-4.35
# df['HSAB'].loc[df['Mname']=='Zn']=np.nan
# df['HSAB'].loc[df['Mname']=='Cd']=-2.04
# df['HSAB'].loc[df['Mname']=='Hg']=-4.64

df['Dissociation']='False'
df['Dissociation'].loc[df['Mname']=='Li']='True'
df['Dissociation'].loc[df['Mname']=='Na']='False'
df['Dissociation'].loc[df['Mname']=='K']='False'
df['Dissociation'].loc[df['Mname']=='Rb']='False'
df['Dissociation'].loc[df['Mname']=='Cs']='False'
df['Dissociation'].loc[df['Mname']=='Be']='True'
df['Dissociation'].loc[df['Mname']=='Mg']='True'
df['Dissociation'].loc[df['Mname']=='Ca']='True'
df['Dissociation'].loc[df['Mname']=='Sr']='False'
df['Dissociation'].loc[df['Mname']=='Ba']='False'
df['Dissociation'].loc[df['Mname']=='Sr']='False'
df['Dissociation'].loc[df['Mname']=='Y']='True'
df['Dissociation'].loc[df['Mname']=='Ti']='True'
df['Dissociation'].loc[df['Mname']=='Zr']='True'
df['Dissociation'].loc[df['Mname']=='Hf']='True'
df['Dissociation'].loc[df['Mname']=='V']='True'
df['Dissociation'].loc[df['Mname']=='Nb']='True'
df['Dissociation'].loc[df['Mname']=='Ta']='True'
df['Dissociation'].loc[df['Mname']=='Cr']='True'
df['Dissociation'].loc[df['Mname']=='Mo']='True'
df['Dissociation'].loc[df['Mname']=='W']='True'
df['Dissociation'].loc[df['Mname']=='Mn']='True'
df['Dissociation'].loc[df['Mname']=='Re']='True'
df['Dissociation'].loc[df['Mname']=='Fe']='True'
df['Dissociation'].loc[df['Mname']=='Ru']='True'
df['Dissociation'].loc[df['Mname']=='Os']='False'
df['Dissociation'].loc[df['Mname']=='Co']='True'
df['Dissociation'].loc[df['Mname']=='Rh']='False'
df['Dissociation'].loc[df['Mname']=='Ir']='False'
df['Dissociation'].loc[df['Mname']=='Ni']='True'
df['Dissociation'].loc[df['Mname']=='Pd']='False'
df['Dissociation'].loc[df['Mname']=='Pt']='False'
df['Dissociation'].loc[df['Mname']=='Cu']='False'
df['Dissociation'].loc[df['Mname']=='Ag']='False'
df['Dissociation'].loc[df['Mname']=='Au']='False'
df['Dissociation'].loc[df['Mname']=='Zn']='True'
df['Dissociation'].loc[df['Mname']=='Cd']='False'
df['Dissociation'].loc[df['Mname']=='Hg']='False'
df['Dissociation'].loc[df['Mname']=='Al']='True'
df['Dissociation'].loc[df['Mname']=='B']='True'
df['Dissociation'].loc[df['Mname']=='Ga']='True'
df['Dissociation'].loc[df['Mname']=='In']='True'
df['Dissociation'].loc[df['Mname']=='Tl']='False'


#############################################################
#### Here we can view structures if we need
##################################################################

#atoms=read('../data/bcc_slab_N.db@')
#view(atoms)



#############################################################
#### Print "df" pandas dataframe for romain
##################################################################
df.to_csv('N2RR_project_data.csv')

#############################################################
#### Plotting periodic table, by first making a small frame and saving .csv
##################################################################
#df_small=df[['Mname', 'Nitride Formation Energy']]
#df_small.to_csv('your.csv', index=False, header=False)

# # # https://github.com/arosen93/ptable_trends
from ptable_trends import ptable_plotter
#ptable_plotter("your.csv", cmap="viridis", alpha=1.0, extended=False)
#ptable_plotter("your.csv", cmap="viridis", alpha=0.5, extended=False)


# df_small=df[['Mname', 'N_energy']]
# df_small = df_small.drop(df_small[df_small['Mname']=='Ga'].index)
# df_small = df_small.drop(df_small[df_small['Mname']=='B'].index)
# df_small = df_small.drop(df_small[df_small['Mname']=='Mn'].index)
# df_small = df_small.drop(df_small[df_small['Mname']=='Hg'].index)
# df_small = df_small.drop(df_small[df_small['Mname']=='In'].index)
# df_small = df_small.drop(df_small[df_small['Mname']=='Cr'].index)
# df_small.to_csv('your.csv', index=False, header=False)

# # # # https://github.com/arosen93/ptable_trends
# ptable_plotter("your.csv", cmap="viridis", alpha=1.0, extended=False)

#############################################################
#### We drop some data that are strong outliers in the dataset & save it
##################################################################
df = df.drop(df[df['Mname']=='Mn'].index)
df = df.drop(df[df['Mname']=='Os'].index)
df = df.drop(df[df['Mname']=='B'].index)
df = df.drop(df[df['Mname']=='Cr'].index)

#df.to_csv('dataset.csv', index=False, header=False)
df.to_csv('dataset.csv',index=False)

#df.plot(x='V SHE', y='NH3_reaction', kind='scatter')


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
#### plotting of figures  X3N vs XN3
##################################################################

fig = plt.figure(figsize=(14,7));
ax=fig.gca()
plt.locator_params(axis='x',nbins=5);plt.grid(True)
plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)

for k in range(0,len(df['V SHE'].values)):
    plt.text(df['M3N Formation Energy'].values[k], df['MN3 Formation Energy'].values[k], df['Mname'].values[k], fontsize=size1)
    plt.scatter(df['M3N Formation Energy'].values[k], df['MN3 Formation Energy'].values[k], c='k', s=sdot)

plt.plot([-10,20],[-10,20],'k-')
plt.xlabel(r'M$_3$N Formation Energy [eV]',fontsize=size1)
plt.ylabel(r'MN$_3$ Formation Energy [eV]',fontsize=size1)
plt.xlim([-1.2,6.5])
plt.ylim([-2,17])
plt.savefig('MN3_vs_M3N.png', dpi=400, bbox_inches='tight')


#############################################################
#### plotting of figures  All vs V SHE
#############################################################

for yval, ylabel in zip(Features,Features_label):
    fig = plt.figure(figsize=(14,7));
    ax=fig.gca()
    plt.locator_params(axis='x',nbins=5);plt.grid(True)
    plt.xticks(fontsize = size3); plt.yticks(fontsize = size3)
    
    for k in range(0,len(df[yval].values)):
        plt.text(df['V SHE'].values[k], df[yval].values[k], df['Mname'].values[k], fontsize=size1)
    
        if df['Dissociation'].values[k]=='True':
            p1=plt.scatter(df['V SHE'].values[k], df[yval].values[k], c='b', s=sdot)
        elif df['Dissociation'].values[k]=='False':
            p2=plt.scatter(df['V SHE'].values[k], df[yval].values[k], c='r', s=sdot)
    
    plt.xlabel(r'Standard Reduction Potential [V vs SHE]',fontsize=size1)
    plt.ylabel(ylabel +' [eV]',fontsize=size1)
    plt.xlim([-3.2,1.55])
    l = ax.legend([p1,p2],['N-N cleaved phase','N-N coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0)
    #plt.savefig(yval+'_vs_SHE.png', dpi=400, bbox_inches='tight')


#############################################################
#### PCA analysis
##################################################################

Features.append('Mname')
Features.append('Dissociation')
Features.append('V SHE')

df_small=df[Features[:-1]]
df_small=df_small. dropna()

# Separating out the features
features=Features[:-3]
features_label=Features_label


x = df_small.loc[:, features].values
# Standardizing the features
test = StandardScaler().fit_transform(x)
#from sklearn.decomposition import PCA
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
    elif df_small['Dissociation'].values[k]=='True':
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='False':
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
        #plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features[i][:-7], color = 'g', ha = 'center', va = 'center',fontsize = s3)
        
    elif features[i]=='Nitride Formation Energy' : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.arrow(0, 0, coeff[i,0]*1.5, coeff[i,1]*1.5,color = 'g',alpha = 0.15)
        plt.text(coeff[i,0]* 1.55, coeff[i,1] * 1.55, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)
        #plt.text(coeff[i,0]* 1.55, coeff[i,1] * 1.55, r'Nitride Formation', color = 'g', ha = 'center', va = 'center',fontsize = s3)

    else : 
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'g',alpha = 0.5)
        plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features_label[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)
        #plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

#plt.text(-0.6,0.6,'=%s' % (pca.explained_variance_ratio_.sum()*100).round() +'%', fontsize=20)
plt.xlim([-0.75,0.75])
plt.ylim([-0.75,0.75])
plt.legend([p1,p2],['N-N cleaved phase','N-N coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=3, fontsize=size1-8,markerscale=1.0)
plt.xticks(fontsize=size2)
plt.yticks(fontsize=size2)
plt.xlabel('Principal Component 1 (' +str(np.round(pca.explained_variance_ratio_[0]*100,0))[:2]+' %)',fontsize=size1)
plt.ylabel('Principal Component 2 (' +str(np.round(pca.explained_variance_ratio_[1]*100,0))[:2]+' %)',fontsize=size1)
fig.text(0.02, 0.9, '(a)', ha='center', fontsize=size2+10)
plt.savefig('PCA_features.png', dpi=400, bbox_inches='tight')



#############################################################
#### PCA analysis - formation energies
##################################################################

# Separating out the features

Features_small.append('Mname')
Features_small.append('Dissociation')
Features_small.append('V SHE')


df_small=df[Features_small[:-1]]
df_small=df_small. dropna()

# Separating out the features
features=Features_small[:-3]
features_label=Features_small_label

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
    elif df_small['Dissociation'].values[k]=='True':
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='False':
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
fig.text(0.02, 0.9, '(b)', ha='center', fontsize=size2+10)
plt.savefig('PCA_features_small.png', dpi=400, bbox_inches='tight')



#############################################################
#### PCA analysis - with SHE
##################################################################

df_small=df[Features]
df_small=df_small.dropna()

# Separating out the features
features=Features[:-3]
features_label=Features_label

# Adding SHE
features.append('V SHE')
features_label.append(r'V$_{SHE}$')


x = df_small.loc[:, features].values
# Standardizing the features
test = StandardScaler().fit_transform(x)
#from sklearn.decomposition import PCA
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
    elif df_small['Dissociation'].values[k]=='True':
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='False':
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
        #plt.text(coeff[i,0]* 1.05, coeff[i,1] * 1.05, features[i], color = 'g', ha = 'center', va = 'center',fontsize = s3)

#plt.text(-0.6,0.6,'=%s' % (pca.explained_variance_ratio_.sum()*100).round() +'%', fontsize=20)
plt.xlim([-0.75,0.75])
plt.ylim([-0.75,0.75])
plt.legend([p1,p2],['N-N cleaved phase','N-N coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=3, fontsize=size1-8,markerscale=1.0)
plt.xticks(fontsize=size2)
plt.yticks(fontsize=size2)
plt.xlabel('Principal Component 1 (' +str(np.round(pca.explained_variance_ratio_[0]*100,0))[:2]+' %)',fontsize=size1)
plt.ylabel('Principal Component 2 (' +str(np.round(pca.explained_variance_ratio_[1]*100,0))[:2]+' %)',fontsize=size1)
fig.text(0.02, 0.9, '(c)', ha='center', fontsize=size2+10)
plt.savefig('PCA_features_Vshe.png', dpi=400, bbox_inches='tight')
#plt.savefig('PC_analysis0.png', dpi=900, bbox_inches='tight')



#############################################################
#### PCA analysis - formation energies - with SHE
##################################################################

df_small=df[Features_small]
df_small=df_small. dropna()

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
    elif df_small['Dissociation'].values[k]=='True':
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='False':
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
plt.savefig('PCA_features_small_Vshe.png', dpi=400, bbox_inches='tight')
#plt.savefig('PC_analysis0.png', dpi=900, bbox_inches='tight')


fig=plt.figure(figsize = (8,8))
plt.locator_params(axis='x',nbins=5);plt.grid(True);plt.locator_params(axis='y',nbins=5)    
for k in range(0,len(x)):
    if text[k,0]=='Li' or text[k,0]=='Ca':
        plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot+300,edgecolors= "black",alpha=1.0,zorder=1)
        plt.text(x[k]*scalex+0.04, y[k]*scaley-0.04, text[k,0], fontsize=size2,zorder=2)
    elif text[k,0]=='Rb' or text[k,0]=='Tl':
        p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
    elif text[k,0]=='Co':
        p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex-0.1, y[k]*scaley, text[k,0], fontsize=size2)
    elif text[k,0]=='Sr' or text[k,0]=='Cd' or text[k,0]=='Ag':
        p2=plt.scatter(x[k]*scalex, y[k]*scaley, color='r',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley-0.05, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='True':
        p1=plt.scatter(x[k]*scalex, y[k]*scaley, color='b',s=sdot,edgecolors= "black",alpha=0.5,zorder=2)
        plt.text(x[k]*scalex, y[k]*scaley, text[k,0], fontsize=size2)
    elif df_small['Dissociation'].values[k]=='False':
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
plt.legend([p1,p2],['Cleaved phase','Coupling phase'],handler_map={tuple: HandlerTuple(ndivide=None)}, loc=4, fontsize=size1-8,markerscale=1.0, columnspacing=0.6, handletextpad=0.05)
plt.xticks(fontsize=size2)
plt.yticks(fontsize=size2)
plt.xlabel('Principal Component 1 (' +str(np.round(pca.explained_variance_ratio_[0]*100,0))[:2]+' %)',fontsize=size1)
plt.ylabel('Principal Component 2 (' +str(np.round(pca.explained_variance_ratio_[1]*100,0))[:2]+' %)',fontsize=size1)
fig.text(0.02, 0.9, '(a)', ha='center', fontsize=size2+10)
plt.savefig('PCA_features_small_Vshe_MS.png', dpi=900, bbox_inches='tight')
#plt.savefig('PC_analysis0.png', dpi=900, bbox_inches='tight')


