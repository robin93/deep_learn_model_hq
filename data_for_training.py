# -*- coding: utf-8 -*-
"""
Created on Mon May  9 13:16:24 2016

@author: Administrator
"""

raw_data_file = 'NASDAQ_62_with_indicators.csv'

import os
os.getcwd()
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')
filepath = os.path.join(cwd,raw_data_file)

raw_data = open(filepath,"r")

row_count = 0
for line in raw_data.readlines():
    if row_count == 0:
        cols = (line.strip()).split(",")
    else:
        break
    row_count += 1

#check the columns and their indexes in the raw data
col_count = 0
for column in cols:
    #print(column,col_count)
    col_count += 1
    
date_OHLC_lag_indexes = [i for i in range(2,28)]
technical_indicator_indexes = [i for i in range(28,49)]
indicator_indexes = [i for i in range(62,78)]
predictor_index = [58]

x_cols = date_OHLC_lag_indexes +technical_indicator_indexes + indicator_indexes
y_cols = predictor_index

all_cols = x_cols + y_cols 

#from numpy import genfromtxt
#raw_data = genfromtxt(filepath,delimiter=',',skip_header = 1)
#print(raw_data.shape)


#division into test and train 
import pandas as pd 
raw_data = pd.read_csv(filepath,header=0,sep=",")
print(raw_data.groupby('Equity').size())
unique_equity = raw_data.Equity.unique()
for equity in unique_equity:
    print(equity)

#treating class imbalance

#writing files for training
