# -*- coding: utf-8 -*-
"""
Created on Sat May 14 07:32:39 2016

@author: Administrator
"""

import os
import pandas as pd

os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')
cwd = os.getcwd()
name = 'Significant_1D_Rise0.5STD_abv_0'
def train_dat():
    path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets'
    file = 'training_data.csv'
    data = pd.read_csv(os.path.join(path ,file) ,sep =',')
    data.drop(data.columns[[0,1,2]], axis=1, inplace=True)
    data[name] = data[name].astype(int)
    data.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets\\Training set for java",'train_data_java.csv'),sep = ',', header=False , index=False)
   
def val_dat():
    path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets'
    file = 'val_data.csv'
    data = pd.read_csv(os.path.join(path ,file) ,sep =',')
    data.drop(data.columns[[0,1,2]], axis=1, inplace=True)
    data[name] = data[name].astype(int)
    data.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets\\Training set for java",'val_data_java.csv'),sep = ',', header=False , index=False)

def test_dat():
    path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets'
    file = 'test.csv'
    data = pd.read_csv(os.path.join(path ,file) ,sep =',')
    data.drop(data.columns[[0,1,2]], axis=1, inplace=True)
    data.drop(data.head(3).index, inplace=True)
    data[name] = data[name].astype(int)
    data.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets\\Training set for java",'test_data_java.csv'),sep = ',', header=False , index=False)


train_dat()
val_dat()
#test_dat()