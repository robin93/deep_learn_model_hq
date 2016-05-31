# -*- coding: utf-8 -*-
"""
Created on Tue May 17 06:51:56 2016

@author: Administrator
"""

import os
import pandas as pd


os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')
cwd = os.getcwd()

path = 'C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\walk_forward_learn_on_16May\\01-05'
file = 'Prediction_01_05_2006_walk_forward.txt'
file1 = 'test_name_date_2006.csv'

data_name = pd.read_csv(os.path.join(path,file1) ,sep =',', header = None) 
data1 = pd.read_csv(os.path.join(path,file) ,sep =',', header = None)

data1[0] = data1[0].apply(lambda s: float(s.strip('[ ')))
data1[1] = data1[0].apply(lambda s: (1-s))
data_name['predictions'] = data1[2]
data_name['conf_0'] = data1[0]
data_name['conf_1'] = data1[1]

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

def hit_rate(data ,col):
    data['Check'] = data.apply(lambda s: True if ( s[2] ==1 and s[col] == 1) else False, axis=1)
    d = data.Check[data.Check == True].count()
#    data.drop('Check', axis =1, inplace = True)
    b = (data[col] == 1).sum()
    return d, b
print(data_name[2][data_name[2]== 1].count())
data_name['predictions_threshold'] = data_name.apply(lambda s: 1 if (s['conf_1'] > 0.7 and s['conf_1'] > s['conf_0']) else 0, axis=1)
print (hit_rate(data_name,'predictions_threshold'))
print (hit_rate(data_name,'predictions'))

main_list_1 = read_filenames_from_symbollist('filenames.txt')
main_list = []
for name in main_list_1 :
    a = ((name.split('#')[0]).split())[3]
    main_list.append(a)

data_main = data_name.loc[data_name[0].isin(main_list)]
print (hit_rate(data_main, 'predictions_threshold'))
print (hit_rate(data_main,'predictions'))