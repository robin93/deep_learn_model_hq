# -*- coding: utf-8 -*-
"""
Created on Mon May  9 05:15:57 2016

@author: Administrator
"""

def read_filenames_from_symbollist(symbol_list_filename):
    import os
    from os import listdir
    from os.path import isfile,join
    os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC')
    cwd = os.getcwd()
    files_list= [f for f in listdir(cwd) if isfile(join(cwd,f))]
    
    
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