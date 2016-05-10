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
raw_data = raw_data[raw_data.columns[all_cols+[1]]]
print(raw_data.groupby('Equity').size())
unique_equity = raw_data.Equity.unique()
train_dataframe,val_dataframe = pd.DataFrame,pd.DataFrame
iter = 0
for equity in unique_equity:
    equity_data = raw_data[(raw_data.Equity==equity)]
    equity_data = equity_data.sort(['Date'])
    equity_train = equity_data.head(len(equity_data)-500)
    equity_val = (equity_data.tail(500)).head(250)
    if iter == 0:
        train_dataframe,val_dataframe = equity_train, equity_val
    else:
        train_dataframe = train_dataframe.append(equity_train)
        val_dataframe = val_dataframe.append(equity_val)
    iter += 1
    print(len(train_dataframe))
    print(len(val_dataframe))

#treating class imbalance

#writing files for training
