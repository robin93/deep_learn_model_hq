# -*- coding: utf-8 -*-
"""
Created on Tue May 10 12:33:16 2016

@author: Administrator
"""

# Check Working Directory
import os
cwd = os.getcwd()

# Set Current Working Directory
os.chdir('C:\\Users\\Administrator\\Desktop\\Data Copy for DL')

cwd = os.getcwd()
# Import Libraries
import lasagne
import numpy as np
import csv as csv
import theano
import theano.tensor as T
from lasagne.nonlinearities import softmax
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.updates import sgd, apply_momentum
from lasagne.objectives import binary_crossentropy,aggregate
from lasagne.init import Constant,Normal
import pandas as pd
import timeit

train_err_file = open("train_error.txt","w+")
val_err_file = open("val_error.txt","w+")
#data = pd.read_csv(os.path.join(cwd ,'55_equity_train_data.csv') ,sep =',', header = None)



def load_data():
    from numpy import genfromtxt
    cols = [i for i in range(0,43)]
    train = genfromtxt('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets\\training_data.csv', delimiter=',',usecols=cols, skip_header=20)
    train  = np.delete(train, np.s_[0:3], axis=1) 
#    train = np.delete(train, np.s_[39:55], axis=1)
    validation = genfromtxt('C:\\Users\\Administrator\\Desktop\\Data Copy for DL\\Training sets\\val_data.csv', delimiter=',',usecols=cols, skip_header=20)
    validation =  np.delete(validation, np.s_[0:3], axis=1)
#    validation = np.delete(validation, np.s_[39:55], axis=1)
#    train = train[0:train.shape[0]/10,:]    
    return train, validation

train , validation = load_data()
train, validation  = train[~np.isnan(train).any(axis=1)],validation[~np.isnan(validation).any(axis=1)]
#train, validation = np.delete(train,np.s_[29:33], axis =1), np.delete(validation,np.s_[29:33], axis=1)         

def iterate_minibatches(inputs,targets,batch_size,shuffle=True):
    assert len(inputs)==len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0,len(inputs)- batch_size+1,batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx+batch_size]
        else:
            excerpt = slice(start_idx,start_idx+batch_size)
        yield inputs[excerpt],targets[excerpt]

def build_mlp(input_var, input_width, output_dim):
    """
    Build a network consistent initial running code of MLP.
    """

    l_in = lasagne.layers.InputLayer((None,input_width),name='INPUT')
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.1)
    l_hid1 = lasagne.layers.DenseLayer(l_in_drop,num_units=100,name = 'Hidden1')
    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.4)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1_drop,num_units=100,name='Hidden2')
    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.4)
    l_out = lasagne.layers.DenseLayer(l_hid2_drop,num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.sigmoid,name = 'OUTPUT')
    return l_out

def run_mlp(train, val, num_epochs):
    # Partition Data
    train_rows, train_cols = train.shape
    train_rows, train_cols = train_rows, (train_cols - 1) 
    val_rows, val_cols = val.shape    
    val_rows, val_cols = val_rows, (val_cols-1)    
    
    X_train,y_train = train[0:train_rows ,0:train_cols],train[0:train_rows,train_cols:]
    X_val,y_val = val[0:val_rows,0:val_cols],val[0:val_rows,val_cols:]
    # Theano variables
    input_var = T.matrix('inputs')
    target_var = T.matrix('targets')
    
    network = build_mlp(input_var, train_cols, 1)
    
#    """loading weight values from the previous model"""
#    with np.load('model_first_run.npz') as f:
#        param_values = [f['arr_%d'%i] for i in range(len(f.files))]
#        param_values[0] = param_values[0][4:43]
#    lasagne.layers.set_all_param_values(network,param_values)
    
    prediction = lasagne.layers.get_output(network,input_var, deterministic = True)
    loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
    loss = aggregate(loss, mode='mean')
    
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.sgd(loss, params, learning_rate=0.5)
    
    # Train Function
    train_fn = theano.function([input_var, target_var],loss, updates=updates)
    
    # Validation function
    val_fn = theano.function([input_var, target_var],loss) 
    
    # test function
    f_test = theano.function([input_var], prediction)    
    
    val_err_list,train_err_list = list(),list()
    print("Starting training...")

    val_err_list,train_err_list = list(),list()
    for epoch in range(num_epochs):
        start_time = timeit.default_timer()        
        train_err = 0
        train_batches = 0
        #start_time = time.time()
        for batch in iterate_minibatches(X_train,y_train,100,shuffle=True):
            inputs,targets = batch
            #print (inputs.shape, targets.shape)
            batch_error = train_fn(inputs,targets)
            #print (list(f_test(inputs)))
            train_err += batch_error
            train_batches += 1
            #print (batch_error, train_batches)
            #if (train_batches%10==0):
                #print(train_batches)
                #print (batch_error)
        train_err_list.append(train_err)    
        
        
        val_err = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 50, shuffle=False):
            inputs, targets = batch
            err = val_fn(inputs, targets)
            val_err += err
            val_batches += 1
            #print (err,val_batches)
        val_err_list.append(val_err)
        # Save results of the epoch
        if (epoch%5 == 0):
            file_name = 'model_epoch_' + str(epoch) + '.npz'
            np.savez(file_name,*lasagne.layers.get_all_param_values(network))
        print ('Epoch: ',epoch, '  train error: ',train_err,' val erro:',val_err)
        train_err_line = (str(epoch)+","+str(train_err)+"\n")
        val_err_line = (str(epoch)+","+str(val_err)+"\n")
        train_err_file.writelines(train_err_line)
        val_err_file.writelines(val_err_line)
        end_time = timeit.default_timer()
        print (end_time - start_time)
    return val_err_list,train_err_list   

    
#def main():•
#    train , test = load_data()
#    print (train[10:60,].shape)
#    val_err_list,train_err_list = run_mlp(train , test, 5)


val_err_list,train_err_list = run_mlp(train , validation, 1000)

#end_time = timeit.default_timer()
#
#time=  (end_time - start_time) / 60
#
#if __name__ == '__main__':
#   main()      
