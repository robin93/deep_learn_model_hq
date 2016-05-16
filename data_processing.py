# -*- coding: utf-8 -*-
"""
Created on Sat May  7 07:26:22 2016

@author: Administrator
"""

"""
   This program is written to read the representative equities of NASDAQ (56 in Number)
   stored in the file named  "symbol_list_NASDAQ_62.txt". Several functions are applied 
   to this data to transform it for further use.
   
   Functions in the file:
   1. read_filenames_from_symbollist
   2. filenames
   3. write_data
   4. add_lags
   5. indicators
   6. cols_to_norm
   7. data_sacntity_check
   8. todate
   9. dateformat
   10. rise_fall
   11. data_mod
   12. main
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd
import os
import pdb
import math
import datetime
# Set current working directory
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')   

# Check working directory
cwd = os.getcwd()

import technical_indicators as ti

def read_filenames_from_symbollist(symbol_list_filename):
    import os
    from os import listdir
    from os.path import isfile,join
    path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC'

    files_list= [f for f in listdir(path) if isfile(join(path,f))]
    
    path_to_symbol_list = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\'
    filename = join(path_to_symbol_list,'symbol_list_NASDAQ_62.txt')
    symbol_list_file = open(filename, 'r')
    symbol_list = symbol_list_file.read().split(',')
    count = 0
    list_of_relevant_filenames = list()
    for symbol in symbol_list:
        for filename in files_list:
            filename_head = filename.split('#')[0]
            equity_symbol = filename_head.split()
            if len(equity_symbol) == 4:
                if symbol == equity_symbol[3]:
                    count += 1
                    list_of_relevant_filenames.append(filename)
    return list_of_relevant_filenames
   
def filenames(file):
    # List of files of equities
    a = list()
    for line in open(file):
        a.append(line.strip())
    for num in range(len(a)): # strip the '\n' from all the elements of the list
        a[num] = a[num].strip()
    return a

def write_data(list):
    filepath = os.path.join('C:\\Users\\Administrator\\Desktop\\Data Copy for DL', 'NASDAQ_62.csv')
    fout = open(filepath,"w")
    count = 0    
    for name in list:
        data = data_mod(name)
        print(data.isnull().sum().sum())
        data.fillna(0)
        if (data.isnull().any().any() == False):
            data.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Processed Data",name),sep = ',') 
        if (count == 0):
            if (data.isnull().any().any() == False): 
                data.to_csv(fout, header = True)
            count = count + 1
        else:
            if (data.isnull().any().any() == False):
                data.to_csv(fout, header = False)
    fout.close()

def add_lags(data_subset):
    data_subset.insert(2,'Open_lag1',data_subset['Open'].shift(+1))
    data_subset.insert(3,'Open_lag2',data_subset['Open'].shift(+2))
    data_subset.insert(4,'Open_lag3',data_subset['Open'].shift(+3))
    data_subset.insert(5,'Open_lag4',data_subset['Open'].shift(+4))
    data_subset.insert(7,'High_lag1',data_subset['High'].shift(+1))
    data_subset.insert(8,'High_lag2',data_subset['High'].shift(+2))
    data_subset.insert(9,'High_lag3',data_subset['High'].shift(+3))
    data_subset.insert(10,'High_lag4',data_subset['High'].shift(+4))
    data_subset.insert(12,'Low_lag1',data_subset['Low'].shift(+1))
    data_subset.insert(13,'Low_lag2',data_subset['Low'].shift(+2))
    data_subset.insert(14,'Low_lag3',data_subset['Low'].shift(+3))
    data_subset.insert(15,'Low_lag4',data_subset['Low'].shift(+4))
    data_subset.insert(17,'Close_lag1',data_subset['Close'].shift(+1))
    data_subset.insert(18,'Close_lag2',data_subset['Close'].shift(+2))
    data_subset.insert(19,'Close_lag3',data_subset['Close'].shift(+3))
    data_subset.insert(20,'Close_lag4',data_subset['Close'].shift(+4))
    data_subset.insert(22,'Volume_lag1',data_subset['Volume'].shift(+1))
    data_subset.insert(23,'Volume_lag2',data_subset['Volume'].shift(+2))
    data_subset.insert(24,'Volume_lag3',data_subset['Volume'].shift(+3))
    data_subset.insert(25,'Volume_lag4',data_subset['Volume'].shift(+4))
    data_subset = data_subset.fillna(method="backfill")
    data_subset = data_subset.fillna(method="ffill")
    return data_subset

"""Indicators that were not taken due to high correlation.
        'MA(5)', 'MA(20)', 'EWMA(9)', 'EWMA(20)', 'EWMA(45)',
       'Volatility(125)','Histogram2', 'RSI(42)','FastStochastic(k%)', 'CCI', 
       'FastStochastic(d%)', 'SlowStochastic(d%)', 'Momentum(7)','MovingVariance',
        'CCI2', 'DisparityIndex(20)','Momentum(20)', 'RateOfChange(20)', 
        'Momentum(40)', 'RateOfChange(40)','MACD2', 'SIGNAL2', 'RateOfChange(7)'
"""
    
def indicators(data_subset):
    data_subset['MA(45)'] = ti.ma(data_subset['Close'],45)
    data_subset['MACD'] = ti.MACD(data_subset['Close'], 12 , 26)
    data_subset['SIGNAL'] = ti.signal(data_subset['Close'],12,26,9)       
    data_subset['Histogram'] = ti.histogram (data_subset['Close'],12,26,9)
    data_subset['RSI(14)'] = ti.rsi(data_subset['Close'], 14)
    data_subset['BolBandUpper'] = ti.ma(data_subset['Close'],20) + (2 * ti.stdev(data_subset['Close'], 20))
    data_subset['BolBandLower'] = ti.ma(data_subset['Close'], 20) - (2 * ti.stdev(data_subset['Close'],20))
    data_subset['MovingVariance'] = ti.MovingVariance(data_subset['Close'],7)
    data_subset['Chaikin'] = ti.Chaikin(data_subset['Close'],data_subset['High'],data_subset['Low'],data_subset['Volume'])
    data_subset['DisparityIndex(10)'] = ti.DisparityIndex(data_subset['Close'],10)
    data_subset['Volatility(30)'] = ti.volatility(data_subset['Close'],30)    
    if (len(data_subset.index) <250):
        data_subset['Volatility(250)'] = 0
    else: 
        data_subset['Volatility(250)'] = ti.volatility(data_subset['Close'],250)
    data_subset['WilliamR(10)'] = ti.WilliamR(data_subset['Close'],data_subset['High'],data_subset['Low'],10)  
    return data_subset
    
def cols_to_norm(data):
    cols_to_norm = ['Open', 'Open_lag1', 'Open_lag2', 'Open_lag3',
       'Open_lag4', 'High', 'High_lag1', 'High_lag2', 'High_lag3',
       'High_lag4', 'Low', 'Low_lag1', 'Low_lag2', 'Low_lag3', 'Low_lag4',
       'Close', 'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4',
       'Volume', 'Volume_lag1', 'Volume_lag2', 'Volume_lag3',
       'Volume_lag4', 'MA(45)', 'MACD', 'SIGNAL',
       'Histogram', 'RSI(14)', 'BolBandUpper', 'BolBandLower',
        'MovingVariance', 'Chaikin', 'DisparityIndex(10)', 
        'Volatility(30)', 'Volatility(250)','WilliamR(10)']
    data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min())) 
    data[cols_to_norm].fillna(0)
    return data
   
def data_sacntity_check(raw_data):

    data = raw_data[(raw_data.High >= raw_data.Low)]
    data = data[(data.Close>=0)&(data.High>=0)&(data.Low>=0)]

    data = data.fillna(method="backfill")
    data = data.fillna(method="ffill")
    return data

def todate(date):
    year = math.floor(date/10000)    
    month = math.floor((date%10000)/100)
    day = date%100
    return datetime.date(year, month, day)

def date_format(data):
    data['Date'] = data['Date'].apply(lambda s: todate(s))
    return data

def PCT_CHG(data, dur):
    name = 'PCT_CHG_' + str(dur)
    data[name] = (data['Close'].shift(-1*dur) - data['Close']) / data['Close'] # Percentage Change 
    data = data.fillna(method="ffill")
    return data

def rise_fall(data,factor,dur):
    name = 'PCT_CHG_' + str(dur)
    mean,std = data[name].mean(), data[name].std() # Mean and Standard Deviation of Percent Change
    up_bound,lower_bound = (mean + (factor*std)), (mean - (factor*std)) # Upper and Lower Bound of Percent Change
    rise =  'Significant_' + str(dur) + 'D_Rise' + str(factor) + 'STD'
    fall =  'Significant_' + str(dur) + 'D_Fall' + str(factor) + 'STD'
    data[rise] = data[name].apply(lambda row : 1 if (row>up_bound) else 0) # Singificant Rise
    data[fall] = data[name].apply(lambda row : 1 if (row<lower_bound) else 0) # Singificant Fall
    data = data.fillna(method="ffill")
    return data
    
def rise_abv_0(data, factor, dur):
    name = 'PCT_CHG_' + str(dur)
    mean,std = data[name].mean(), data[name].std() # Mean and Standard Deviation of Percent Change
    up_bound,lower_bound = (factor*std), (factor*std) # Upper and Lower Bound of Percent Change
    rise =  'Significant_' + str(dur) + 'D_Rise' + str(factor) + 'STD_abv_0'
    fall =  'Significant_' + str(dur) + 'D_Fall' + str(factor) + 'STD_abv_0'
    data[rise] = data[name].apply(lambda row : 1 if (row>up_bound) else 0) # Singificant Rise
    data[fall] = data[name].apply(lambda row : 1 if (row<lower_bound) else 0) # Singificant Fall
    data = data.fillna(method="ffill")
    return data

#==============================================================================
#path = "C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC"
#file = 'AMEX EQUITIES FUT ABE#6531 1D.csv'
#dat = data = pd.read_csv(os.path.join(path ,file) ,sep =',', header=None)
#dat.drop(dat.columns[[1]], axis=1, inplace = True)
#dat.drop(dat.columns[6:], axis=1, inplace = True)
#dat.columns = ['Date', 'Open', 'High', 'Low', 'Close','Volume']
#rise_fall(dat,0.5,1)
#dat['Check'] = dat.apply(lambda s: True if ( s['Significant_1D_Rise0.5STD'] ==1 and s['PCT_CHG']<0) else False, axis=1)
#any(dat.Check == True)
#==============================================================================
    
def data_mod(file):
    name =   file.strip('.csv') 
    path = "C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC"
    data = pd.read_csv(os.path.join(path ,file) ,sep =',', header=None)
    data.drop(data.columns[[1]], axis=1, inplace = True)
    data.drop(data.columns[6:], axis=1, inplace = True)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close','Volume']
    data = data_sacntity_check(data)
    data = date_format(data)
    data = add_lags(data) 
    data = indicators(data)
    data = PCT_CHG(data, 1)
    data = PCT_CHG(data, 3)
#    data = rise_abv_0(data, 1.5,1)
#    data = rise_abv_0(data,1.5,3)
#    data = rise_abv_0(data,1.0,1)
#    data = rise_abv_0(data,1.0,3)
#    data = rise_abv_0(data,0.5,1)
#    data = rise_abv_0(data,0.5,3)
    
    data = rise_fall(data,1.5,1)
    data = rise_fall(data,1.5,3)
    data = rise_fall(data,1.0,1)
    data = rise_fall(data,1.0,3)
    data = rise_fall(data,0.5,1)
    data = rise_fall(data,0.5,3)    
    data = cols_to_norm(data)
    equity_name = ((name.split('#')[0]).split())[3]
    data.insert(0, 'Equity',equity_name)    
    return data

def main():
    filename = read_filenames_from_symbollist('filenames.txt')
    # choose the equities from filename list to be analysed 
    write_data(filename) # File will be written by the name of 'out.csv' to the working directory

if __name__ == '__main__':
    main() 
