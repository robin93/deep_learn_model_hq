# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
def setwd(path):
    # Check working directory
    import os 
    os.getcwd()
    
    # Set current working directory
    os.chdir(path)  

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
    fout = open("out.csv","a")
    for name in list:
        for line in open(name):            
            fout.write(line)
    fout.close()

def main():
    setwd('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\CC')
    filename = filenames('filenames.txt')
    # choose the equities from filename list to be analysed   
    b = filename[0:1]
    build_data_file(b) # FIle will be written by the name of 'out.csv' to the working directory
    
if __name__ == '__main__':
    main()    
