# -*- coding: utf-8 -*-
"""
Created on Sat May  7 07:26:22 2016

@author: Administrator
"""

import numpy as np
from numpy import genfromtxt
import pandas as pd
import os

# Set current working directory
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC')   

# Check working directory
cwd = os.getcwd()

import technical_indicators as ti

   
def filenames(file):
    # List of files of equities
    a = list()
    for line in open(file):
        a.append(line.strip())
    for num in range(len(a)): # strip the '\n' from all the elements of the list
        a[num] = a[num].strip()
    return a

def write_data(list):
    for name in list:
        data = data_mod(name)
        data.to_csv(os.path.join("C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Processed Data",name),sep = ',') 
            
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
    
def indicators(data_subset):
    data_subset['MA(5)'] = ti.ma(data_subset['Close'],5)
    data_subset['MA(20)'] = ti.ma(data_subset['Close'],20)
    data_subset['EWMA(20)'] = ti.ewma(data_subset['Close'], 9)
    data_subset['MACD'] = ti.MACD(data_subset['Close'], 12 , 26)
    data_subset['SIGNAL'] = ti.signal(data_subset['Close'],12,26,9)
    data_subset['Histogram'] = ti.histogram (data_subset['Close'],12,26,9)
    data_subset['RSI(14)'] = ti.rsi(data_subset['Close'], 14)
    data_subset['BolBandUpper'] = ti.ma(data_subset['Close'],20) + (2 * ti.stdev(data_subset['Close'], 20))
    data_subset['BolBandLower'] = ti.ma(data_subset['Close'], 20) - (2 * ti.stdev(data_subset['Close'],20))
    data_subset['FastStochastic(k%)'],data_subset['FastStochastic(d%)'] , data_subset['SlowStochastic(d%)']= ti.STOK(data_subset['Close'],data_subset['High'],data_subset['Low'],14,3)
    data_subset['Momentum(7)'] = ti.Momentum(data_subset['Close'],7)
    data_subset['RateOfChange(7)'] = ti.RateOfChange(data_subset['Close'],7)
    data_subset['MovingVariance'] = ti.MovingVariance(data_subset['Close'],7)
    data_subset['CCI'] = ti.CCI(data_subset['Close'],data_subset['High'],data_subset['Low'],20,20)
    data_subset['Chaikin'] = ti.Chaikin(data_subset['Close'],data_subset['High'],data_subset['Low'],data_subset['Volume'])
    data_subset['DisparityIndex(10)'] = ti.DisparityIndex(data_subset['Close'],10)
    data_subset['Volatility(20)'] = ti.volatility(data_subset['Close'],20)
    data_subset['Volatility(10)'] = ti.volatility(data_subset['Close'],10)
    data_subset['WilliamR(10)'] = ti.WilliamR(data_subset['Close'],data_subset['High'],data_subset['Low'],10)
    return data_subset

#def params(data):
#    mean =   data["Close"].mean()  
#    std = data["Close"].std()
#    return mean, std

def rise_fall(data):
    data['one_day_pct_change'] = ((data['Close']-data['Open'])*100)/data['Open']
    return data
    
    

def cols_to_norm(data):
    cols_to_norm = ['Open', 'Open_lag1', 'Open_lag2', 'Open_lag3',
       'Open_lag4', 'High', 'High_lag1', 'High_lag2', 'High_lag3',
       'High_lag4', 'Low', 'Low_lag1', 'Low_lag2', 'Low_lag3', 'Low_lag4',
       'Close', 'Close_lag1', 'Close_lag2', 'Close_lag3', 'Close_lag4',
       'Volume', 'Volume_lag1', 'Volume_lag2', 'Volume_lag3',
       'Volume_lag4', 'MA(5)', 'MA(20)', 'EWMA(20)', 'MACD', 'SIGNAL',
       'Histogram', 'RSI(14)', 'BolBandUpper', 'BolBandLower',
       'FastStochastic(k%)', 'FastStochastic(d%)', 'SlowStochastic(d%)',
       'Momentum(7)', 'RateOfChange(7)', 'MovingVariance', 'CCI',
       'Chaikin', 'DisparityIndex(10)', 'Volatility(20)', 'Volatility(10)',
       'WilliamR(10)']
    data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min())) 
    return data
   
def data_sacntity_check(raw_data):

    data = raw_data[(raw_data.High >= raw_data.Low)]
    data = data[(data.Close>=0)&(data.High>=0)&(data.Low>=0)]

    data = data.fillna(method="backfill")
    data = data.fillna(method="ffill")
    return data
    
def data_mod(file):
    name =   file.strip('.csv')  
    data = pd.read_csv(os.path.join(cwd ,file) ,sep =',', header=None)
    data.drop(data.columns[[1]], axis=1, inplace = True)
    data.drop(data.columns[6:], axis=1, inplace = True)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close','Volume']
    data = data_sacntity_check(data)
    data = add_lags(data)  
    data = indicators(data)
    data = rise_fall(data)
    data = cols_to_norm(data)
    data.insert(0, 'Equity', name)    
    return data

def main():
    filename = filenames('filenames.txt')
    # choose the equities from filename list to be analysed   
    b = filename[0:1]
    write_data(b) # FIle will be written by the name of 'out.csv' to the working directory

if __name__ == '__main__':
    main() 
