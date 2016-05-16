# -*- coding: utf-8 -*-
"""
Created on Wed May 11 08:00:45 2016

@author: Administrator
"""

# Check working directory
import os
os.getcwd()

# Set current working directory
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Processed Data')  

def filenames(file):
    # List of files of equities
    a = list()
    for line in open(file):
        a.append(line.strip())
    for num in range(len(a)): # strip the '\n' from all the elements of the list
        a[num] = a[num].strip()
    return a
    
def build_data_file(list):    
    # Writing the data of files to a single csv
    filepath = os.path.join('C:\\Users\\Administrator\\Desktop\\Data Copy for DL', 'NASDAQ_62.csv')
    fout = open(filepath,"w")
    count = 0    
    for name in list:       
        if (count == 0):       
            f = open(name)
            for line in f:
                fout.write(line)
            count = count + 1
        else:
            f = open(name)
            c = 0
            for line in f:
                if (c == 0 and count ==1):
                    c = c+1
                else:
                    fout.write(line)
        

    fout.close()
    

def industry_indicator_function(row):
    equity = row['Equity']
    if (symbol_ind_dict[equity] == industry):
        return 1
    else:
        return 0

def capsize_indicator_function(row):
    equity = row['Equity']
    if (symbol_capsize_dict[equity] == capsize):
        return 1
    else:
        return 0


def add_industry_size_indicators():
    ind_list = ["Basic Industries","Capital Goods","Consumer Durables","Consumer Non-Dur.","Consumer Services","Energy","Finance","Healthcare","Public Utilities","Technology","Transportation"]
    capsize_list = ["Mid-cap","Small-cap","Micro-cap","Large-cap","Mega-cap"]    
    import pandas as pd
    os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')
    cwd = os.getcwd()
    combined_data = pd.read_csv(os.path.join(cwd,'NASDAQ_62.csv'),header=0,sep=",")
    lookup_file = open(os.path.join(cwd,'symbol_ind_capsize.txt'),"r")
    symbol_ind_dict = dict()
    symbol_capsize_dict = dict()
    for line in lookup_file.readlines():
        symbol = (line.split(","))[0]
        Industry = (line.split(","))[1]
        Capsize = (line.split(","))[2].strip()
        if symbol not in symbol_ind_dict.keys():
            symbol_ind_dict[symbol] = Industry
        if symbol not in symbol_capsize_dict.keys():
            symbol_capsize_dict[symbol] = Capsize
    def industry_indicator_function(row):
        equity = row['Equity']
        if (symbol_ind_dict[equity] == industry):
            return 1
        else:
            return 0
    def capsize_indicator_function(row):
        equity = row['Equity']
        if (symbol_capsize_dict[equity] == capsize):
            return 1
        else:
            return 0
    for industry in ind_list:
        combined_data[industry] = combined_data.apply(industry_indicator_function,axis=1)
        print(industry)
    for capsize in capsize_list:
        combined_data[capsize] = combined_data.apply(capsize_indicator_function,axis=1)
        print(capsize)
    combined_data.to_csv('NASDAQ_62_with_indicators.csv',sep=',',index=False)
    
    
    

def main():
    import os
    from os import listdir
    from os.path import isfile,join
    os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Processed Data')
    cwd = os.getcwd()
    files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
    files_list = [ x for x in files_list if ".csv" in x ]     
    print (files_list)
    print (len(files_list))
    build_data_file(files_list) # File will be written by the name of 'out.csv' to the working directory
    #add_industry_size_indicators()
    
    
if __name__ == '__main__':
    main()    

import pandas as pd
import os
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')
cwd = os.getcwd()
data = pd.read_csv(os.path.join(cwd,'NASDAQ_62.csv'),header=0,sep=",")
#data.columns.values
b = ['Unnamed: 0', 'Equity', 'Date', 'Open', 'Open_lag1', 'High_lag1',
       'Low_lag1', 'Close_lag1', 'Volume_lag1', 'Open_lag2', 'High_lag2',
       'Low_lag2', 'Close_lag2', 'Volume_lag2', 'Open_lag3', 'High_lag3',
       'Low_lag3', 'Close_lag3', 'Volume_lag3', 'Open_lag4', 'High_lag4',
       'Low_lag4', 'Close_lag4', 'Volume_lag4', 'Open_lag5', 'High_lag5',
       'Low_lag5', 'Close_lag5', 'Volume_lag5', 'High', 'Low', 'Close',
       'Volume','MA(5)', 'MA(20)', 'EWMA(9)', 'EWMA(20)', 'EWMA(45)',
       'Volatility(125)','Histogram2', 'RSI(42)','FastStochastic(k%)', 'CCI', 
       'FastStochastic(d%)', 'SlowStochastic(d%)', 'Momentum(7)','MovingVariance',
        'CCI2', 'DisparityIndex(20)','Momentum(20)', 'RateOfChange(20)', 
        'Momentum(40)', 'RateOfChange(40)','MACD2', 'SIGNAL2', 'RateOfChange(7)']
data.drop(b, axis=1, inplace=True) 
data.columns.values      
corr = data.corr()
corr.to_csv('correlations_selected.csv', sep = ',')
""" """