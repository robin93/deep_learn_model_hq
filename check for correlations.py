# -*- coding: utf-8 -*-
"""
Created on Wed May 11 07:55:02 2016

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
import read_filenames as rf

   
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
    for i in range(5):
        open = 'Open_lag' + str(i+1)
        high = 'High_lag' + str(i+1) 
        low = 'Low_lag' + str(i+1)
        close = 'Close_lag' + str(i+1) 
        vol = 'Volume_lag' + str(i+1) 
        
        move = i+1        
        pos = 2 + (i*5)        
        
        data_subset.insert(pos,open,data_subset['Open'].shift(+move))
        data_subset.insert((pos+1),high,data_subset['High'].shift(+move))
        data_subset.insert((pos+2),low,data_subset['Low'].shift(+move))
        data_subset.insert((pos+3),close,data_subset['Close'].shift(+move))
        data_subset.insert((pos+4),vol,data_subset['Volume'].shift(+move))                        
            
    data_subset = data_subset.fillna(method="backfill")
    data_subset = data_subset.fillna(method="ffill")
    return data_subset
    
def indicators(data_subset):
    data_subset['MA(5)'] = ti.ma(data_subset['Close'],5)
    data_subset['MA(20)'] = ti.ma(data_subset['Close'],20)
    data_subset['MA(45)'] = ti.ma(data_subset['Close'],45)
    data_subset['EWMA(9)'] = ti.ewma(data_subset['Close'], 9)
    data_subset['EWMA(20)'] = ti.ewma(data_subset['Close'], 20)
    data_subset['EWMA(45)'] = ti.ewma(data_subset['Close'], 45)
    
    data_subset['MACD1'] = ti.MACD(data_subset['Close'], 12 , 26)
    data_subset['SIGNAL1'] = ti.signal(data_subset['Close'],12,26,9)
    data_subset['Histogram1'] = ti.histogram (data_subset['Close'],12,26,9)
    
    data_subset['MACD2'] = ti.MACD(data_subset['Close'], 15 , 45)
    data_subset['SIGNAL2'] = ti.signal(data_subset['Close'],15,45,15)
    data_subset['Histogram2'] = ti.histogram (data_subset['Close'],15,45,15)
    
    data_subset['RSI(14)'] = ti.rsi(data_subset['Close'], 14)
    data_subset['RSI(42)'] = ti.rsi(data_subset['Close'], 42)    
    
    data_subset['BolBandUpper'] = ti.ma(data_subset['Close'],20) + (2 * ti.stdev(data_subset['Close'], 20))
    data_subset['BolBandLower'] = ti.ma(data_subset['Close'], 20) - (2 * ti.stdev(data_subset['Close'],20))
    
    data_subset['FastStochastic(k%)'],data_subset['FastStochastic(d%)'] , data_subset['SlowStochastic(d%)']= ti.STOK(data_subset['Close'],data_subset['High'],data_subset['Low'],14,3)
    
    data_subset['Momentum(7)'] = ti.Momentum(data_subset['Close'],7)
    data_subset['RateOfChange(7)'] = ti.RateOfChange(data_subset['Close'],7)
    
    data_subset['Momentum(20)'] = ti.Momentum(data_subset['Close'],20)
    data_subset['RateOfChange(20)'] = ti.RateOfChange(data_subset['Close'],20)  
    
    data_subset['Momentum(40)'] = ti.Momentum(data_subset['Close'],40)
    data_subset['RateOfChange(40)'] = ti.RateOfChange(data_subset['Close'],40)    

    data_subset['MovingVariance'] = ti.MovingVariance(data_subset['Close'],7)
    data_subset['MovingVariance20'] = ti.MovingVariance(data_subset['Close'],20)

    data_subset['CCI'] = ti.CCI(data_subset['Close'],data_subset['High'],data_subset['Low'],20,20)
    data_subset['CCI2'] = ti.CCI(data_subset['Close'],data_subset['High'],data_subset['Low'],20,20)


    data_subset['Chaikin'] = ti.Chaikin(data_subset['Close'],data_subset['High'],data_subset['Low'],data_subset['Volume'])

    data_subset['DisparityIndex(10)'] = ti.DisparityIndex(data_subset['Close'],10)
    data_subset['DisparityIndex(20)'] = ti.DisparityIndex(data_subset['Close'],20)
    data_subset['DisparityIndex(20)'] = ti.DisparityIndex(data_subset['Close'],40)

    data_subset['Volatility(30)'] = ti.volatility(data_subset['Close'],30)
    data_subset['Volatility(125)'] = ti.volatility(data_subset['Close'],125)
    data_subset['Volatility(250)'] = ti.volatility(data_subset['Close'],250)

    data_subset['WilliamR(10)'] = ti.WilliamR(data_subset['Close'],data_subset['High'],data_subset['Low'],10)
    
    return data_subset.corr()

#def rise_fall(data):
#    data['one_day_pct_change'] = ((data['Close']-data['Open'])*100)/data['Open']
#    return data
    
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

def rise_fall(data,factor,dur):
    data['PCT_CHG'] = (data['Close'].shift(-1*dur) - data['Close']) / data['Close'] # Percentage Change
    mean,std = data['PCT_CHG'].mean(), data['PCT_CHG'].std() # Mean and Standard Deviation of Percent Change
    up_bound,lower_bound = mean + factor*std, mean - factor*std # Upper and Lower Bound of Percent Change
    rise =  'Significant_' + str(dur) + 'D_Rise' + str(factor) + 'STD'
    fall =  'Significant_' + str(dur) + 'D_Fall' + str(factor) + 'STD'
    data[rise] = data['PCT_CHG'].apply(lambda row : 1 if (row>up_bound) else 0).shift(-1) # Singificant Rise
    data[fall] = data['PCT_CHG'].apply(lambda row : 1 if (row<lower_bound) else 0).shift(-1) # Singificant Fall
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
#    data = rise_fall(data,1.5,1)
#    data = rise_fall(data,1.5,3)
#    data = rise_fall(data,1.0,1)
#    data = rise_fall(data,1.0,3)
#    data = rise_fall(data,0.5,1)
#    data = rise_fall(data,0.5,3)    
#    data = cols_to_norm(data)
    equity_name = ((name.split('#')[0]).split())[3]
    data.insert(0, 'Equity',equity_name)    
    return data

def main():
    filename = rf.read_filenames_from_symbollist('filenames.txt')
    # choose the equities from filename list to be analysed   
    b = filename
    write_data(b) # FIle will be written by the name of 'out.csv' to the working directory

if __name__ == '__main__':
    main() 
