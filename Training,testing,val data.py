# -*- coding: utf-8 -*-
"""
Created on Fri May 13 05:48:58 2016

@author: Administrator
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
import pdb
import math
import datetime
import csv
# Set current working directory
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')   

# Check working directory
cwd = os.getcwd()


import technical_indicators as ti
import read_filenames as rf
import data_processing as dp

path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL'
file = 'NASDAQ_62.csv'
data = pd.read_csv(os.path.join(path ,file) ,sep =',')
data.drop(data.columns[[0]], axis=1, inplace=True)
data.columns.values
training_start = datetime.date(1995,1,1)
training_end = datetime.date(1999,1,1)
val_start =  datetime.date(1999,1,1)
val_end = datetime.date(2000,1,1)
testing_start = datetime.date(2000,1,1)
testing_end = datetime.date(2001,1,1)
""" Choose any one to be predicted
   'Significant_1D_Rise0.5STD'
   'Significant_1D_Rise1.0STD'
   'Significant_1D_Rise1.5STD'
   'Significant_3D_Rise0.5STD'
   'Significant_3D_Rise1.0STD'
   'Significant_3D_Rise1.5STD'
   'Significant_1D_Fall0.5STD'
   'Significant_1D_Fall1.0STD'
   'Significant_1D_Fall1.5STD'
   'Significant_3D_Fall0.5STD'
   'Significant_3D_Fall1.0STD'
   'Significant_3D_Fall1.5STD'
"""
list_of_predictors =[
                   'Significant_1D_Rise0.5STD',
                   'Significant_1D_Rise1.0STD',
                   'Significant_1D_Rise1.5STD',
                   'Significant_3D_Rise0.5STD',
                   'Significant_3D_Rise1.0STD',
                   'Significant_3D_Rise1.5STD',
                   'Significant_1D_Fall0.5STD',
                   'Significant_1D_Fall1.0STD',
                   'Significant_1D_Fall1.5STD',
                   'Significant_3D_Fall0.5STD',
                   'Significant_3D_Fall1.0STD',
                   'Significant_3D_Fall1.5STD'                    
                    ] 
#prediction_class = 'Significant_1D_Rise0.5STD'
drop_list = list_of_predictors.copy()
#drop_list.remove('Significant_1D_Rise0.5STD')

data.drop(drop_list, inplace=True, axis =1)
    
def rise_abv_0(data, factor, dur):
    name  = 'PCT_CHG_' + str(dur)
    mean,std = data[name].mean(), data[name].std() # Mean and Standard Deviation of Percent Change
    up_bound,lower_bound = (factor*std), (factor*std) # Upper and Lower Bound of Percent Change
    rise =  'Significant_' + str(dur) + 'D_Rise' + str(factor) + 'STD_abv_0'
    fall =  'Significant_' + str(dur) + 'D_Fall' + str(factor) + 'STD_abv_0'
    data[rise] = data[name].apply(lambda row : 1 if (row>up_bound) else 0) # Singificant Rise
    data[fall] = data[name].apply(lambda row : 1 if (row<lower_bound) else 0) # Singificant Fall
    data = data.fillna(method="ffill")
    return data
# Adding rise above 0 in data
data = rise_abv_0(data, 0.5,1)
data.drop('Significant_1D_Fall0.5STD_abv_0', inplace =True, axis=1)

##### Part for performing checks on data

#data['Check'] = data.apply(lambda s: True if ( s['Significant_1D_Rise0.5STD'] ==1 and s['PCT_CHG']<0) else False, axis=1)
#data.Check[data.Check == True].count()
#(data['Significant_1D_Rise0.5STD'] == 1).sum()

def training_data(data, training_start, training_end,drop_list):
    list_of_equities = pd.unique(data.Equity.ravel())
    data['Date'] = pd.to_datetime(data['Date'])
    mask = (data['Date'] >= training_start) & (data['Date'] <= training_end)
    training_data = data.loc[mask]
    return training_data

def val_data(data, val_start, val_end,drop_list):
    list_of_equities = pd.unique(data.Equity.ravel())
    data['Date'] = pd.to_datetime(data['Date'])
    mask = (data['Date'] >= val_start) & (data['Date'] <= val_end)
    val_data = data.loc[mask]
    return val_data
    
def testing_data(data, testing_start, testing_end,drop_list):
    list_of_equities = pd.unique(data.Equity.ravel())
    data['Date'] = pd.to_datetime(data['Date'])
    mask = (data['Date'] >= testing_start) & (data['Date'] <= testing_end)
    testing_data = data.loc[mask]
    return testing_data

def todate(date):
    year = math.floor(date/10000)    
    month = math.floor((date%10000)/100)
    day = date%100
    return datetime.date(year, month, day)

def date_format(data):
    data['Date'] = data['Date'].apply(lambda s: todate(s))
    return data

def read_file_NASDAQ():
    from os import listdir
    from os.path import isfile,join
    file_path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC'
    files_list= [f for f in listdir(file_path) if isfile(join(file_path,f))]

    list_of_relevant_filenames = list()
    
    for filename in files_list:
        filename_head = filename.split(' ')[0]
        if (filename_head) == 'NASDAQ':
            list_of_relevant_filenames.append(filename)
    
    return list_of_relevant_filenames



def data_mod(file):
    name =   file.strip('.csv') 
    file_path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC'
    data = pd.read_csv(os.path.join(file_path ,file) ,sep =',', header=None)
    if (len(data.index) == 0):
        return 0
    else:    
        data.drop(data.columns[[1]], axis=1, inplace = True)
        data.drop(data.columns[6:], axis=1, inplace = True)
        data.columns = ['Date', 'Open', 'High', 'Low', 'Close','Volume']
        data = date_format(data)
        data = dp.data_sacntity_check(data)
        data = dp.add_lags(data)
        data = dp.indicators(data)
        data = dp.PCT_CHG(data, 1)
        data = dp.PCT_CHG(data, 3)
        data = dp.rise_fall(data,1.5,1)
        data = dp.rise_fall(data,1.5,3)
        data = dp.rise_fall(data,1.0,1)
        data = dp.rise_fall(data,1.0,3)
        data = dp.rise_fall(data,0.5,1)
        data = dp.rise_fall(data,0.5,3)
        data = dp.rise_abv_0(data,0.5,1)
        data.drop('Significant_1D_Fall0.5STD_abv_0', inplace =True, axis=1)
        data = dp.cols_to_norm(data)
        data['Volatility(250)'] = 0
        equity_name = ((name.split('#')[0]).split())[3]
        data.insert(0, 'Equity',equity_name)    
        return data
   
def test_data(testing_start, testing_end, drop_list):
    list1 = read_file_NASDAQ()    
    filepath = os.path.join('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets', 'test.csv')
    fout = open(filepath,"w")    
    for name in list1:
        data = data_mod(name)
        data.drop(drop_list, inplace=True, axis =1)
        mask = (data['Date'] >= testing_start) & (data['Date'] <= testing_end)
        count = 0
        if (data.isnull().any().any() == False):
            data['Date'] = pd.to_datetime(data['Date'])
            testing_data = data.loc[mask]
            if (count == 0):
                testing_data.to_csv(fout, header = True)
                count = count + 1
            else:
                testing_data.to_csv(fout, header = False)
                count = count + 1
                print (count)
    fout.close()                        

def undersample(data):
    b = data.columns.values
    train_matrix = data.as_matrix()
    unq_rise,unq_idx_rise = np.unique(train_matrix[:,-1],return_inverse=True)
    unq_cnt_rise = np.bincount(unq_idx_rise)
    cnt = np.min(unq_cnt_rise)  #undersampling use min for undersampling, mac for oversampling
    tdata = np.empty((cnt*len(unq_rise),) + train_matrix.shape[1:],train_matrix.dtype)
    for j in range(len(unq_rise)):
        np.random.seed(20*j + 10)
        indices = np.random.choice(np.where(unq_idx_rise==j)[0],cnt)
        tdata[j*cnt:(j+1)*cnt]=train_matrix[indices]
    train_data = tdata[np.argsort(tdata[:,1])]
    data = pd.DataFrame(train_data, columns = b)
    return data

def oversample(data):
    b = data.columns.values
    train_matrix = data.as_matrix()
    unq_rise,unq_idx_rise = np.unique(train_matrix[:,-1],return_inverse=True)
    unq_cnt_rise = np.bincount(unq_idx_rise)
    cnt = np.max(unq_cnt_rise)  #undersampling use min for undersampling, mac for oversampling
    tdata = np.empty((cnt*len(unq_rise),) + train_matrix.shape[1:],train_matrix.dtype)
    for j in range(len(unq_rise)):
        np.random.seed(20*j + 10)
        indices = np.random.choice(np.where(unq_idx_rise==j)[0],cnt)
        tdata[j*cnt:(j+1)*cnt]=train_matrix[indices]
    train_data = tdata[np.argsort(tdata[:,1])]
    data = pd.DataFrame(train_data, columns = b)
    return data
    
    
def write_file(data,training_start, training_end,val_start, val_end, testing_start, testing_end ,drop_list):
    training_dat = training_data(data, training_start, training_end,drop_list)
    training_dat = oversample(training_dat)
    val_dat = val_data(data, val_start, val_end,drop_list)
    testing_dat = testing_data(data, testing_start, testing_end,drop_list)
    
    training_dat.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets",'training_data.csv'),sep = ',')
    val_dat.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets",'val_data.csv'),sep = ',')
#    testing_dat.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets",'testing_data.csv'),sep = ',')

def main():
    write_file(data,training_start, training_end,val_start, val_end, testing_start, testing_end ,drop_list)
    test_data(testing_start,testing_end, drop_list)  
    
if __name__ == '__main__':
   main()