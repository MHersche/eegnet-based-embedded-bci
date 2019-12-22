#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
import sys

sys.path.insert(1, '../')

from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
# EEGNet models
import models as models
from sklearn.model_selection import KFold

# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
duration_datagetter = 3 # duration of the events that we get from the data getter
n_classes = 4 # can be 2, 3, 4
channel_size = 64 # look for 64, 32, 16, 8
ds_factor = 4# downsampling factor LOOK For: 2,3,4
PATH = "../../files/"
MODEL_NAME = f'{n_classes}class_{channel_size}ch_freqfactor{ds_factor}'

# current_time = datetime.now()

results_dir=f'results/{MODEL_NAME}'
str_param = f'm1'
param_value = -5
if  not os.path.isdir(f'./{results_dir}'):
    os.mkdir (results_dir)
        

# Load data
# X_all_Train, y_Train = get.get_data(PATH,n_classes = n_classes)
X_all_Train = np.loadtxt('../../signals_csv/data_4class.csv')
y_Train = np.loadtxt('../../signals_csv/data_4class_labels.csv')

old_shape = np.shape(X_all_Train)
X_all_Train = np.reshape(X_all_Train,(old_shape[0],old_shape[1]//480,480))

shape = np.shape(X_all_Train) # shape of the input of data getter

# do downsampling if True
if False:
    ds_samplesize = int(shape[2]/ds_factor) # downsampled size of the sample size

    X_Train_downsampled = np.zeros((shape[0],shape[1],ds_samplesize)) #create new array for downsampling

    # put the downsampled data in it
    for downsampling in range(ds_samplesize):
        X_Train_downsampled[:,:,downsampling] = X_all_Train[:,:,downsampling*ds_factor]
        # X_result = X_Train_downsampled

X_Train_downsampled = X_all_Train
# do channel reduction and duration reduction if True
if True:
    shape = np.shape(X_Train_downsampled)
    channels_reduced_27 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,38,39,40,41,44,45]
    channels_10_20system_19 = [8,10,12,21,23,29,31,33,35,37,40,41,46,48,50,52,54,60,62]
    channels_10_20system_38 = [0,2,4,6,8,10,12,14,16,18,20,21,22,23,24,26,28,29,31,33,35,37,40,41,42,43,46,48,50,52,54,55,57,59,60,61,62,63]
    freq = int(np.shape(X_Train_downsampled)[2]/duration_datagetter)

    X_Train_reduced = np.zeros((shape[0],channel_size,shape[2]))
    X_Train_reduced[:,:,:] = X_Train_downsampled[:,:,:]
    X_result = X_Train_reduced

X_Train = X_result
# Expand dimensions to match expected EEGNet input
X_Train_real = (np.expand_dims(X_Train, axis=1))

# use sample size
SAMPLE_SIZE = np.shape(X_Train_real)[3]

# convert labels to one-hot encodings.
y_Train_cat      = np_utils.to_categorical(y_Train)

# using 5 folds
kf = KFold(n_splits = 5)
# After hyperparameter search: we chose 10**(-4) for the learning rate
alphas = [10**i for i in range(param_value,param_value + 1)]
lr_parameters = [3,4,5]
# results = np.zeros(len(alphas))

# create a 2D array for fold creation. # 640 is here the sample size.
x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], channel_size*SAMPLE_SIZE))


for i in range(len(alphas)):

    training_accuracies = np.array([])
    validation_accuracies = np.array([])
    training_losses = np.array([])
    validation_losses = np.array([])
    epoch_num = 0



    # creating the model every time?
    model = models.EEGNet(nb_classes = n_classes, Chans=channel_size, Samples=SAMPLE_SIZE, regRate=0.25,
                    dropoutRate=0.2, kernLength=128, numFilters=8, dropoutType='Dropout')
    
    # compile the model and set the optimizers - Find optimal learning rate between 10e-3 and 10e0
    adam_alpha = Adam(lr=alphas[i]) # originally: optimizer='adam'
    model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])    

    model.save(f'models/{MODEL_NAME}.h5')
    
    best_valid_accu = 0.
    patience = 200
    no_increase = 0

    # counter for the csv files
    counter = 0

    for train, test in kf.split(x_train_aux):
        if counter > 0:
            continue

        np.random.seed(100)
        np.random.shuffle(train)

        for iii in lr_parameters:

            print("Lr exponent: minus...")
            print(iii)
            if(iii == 3):
                epoch_num = 20
            elif iii == 4:
                epoch_num = 30
            elif iii == 5:
                epoch_num = 50

            model = load_model(f'models/{MODEL_NAME}.h5')

            adam_alpha = Adam(lr=10**(-iii))
            model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])  


            # creating a history object
            history = model.fit(X_Train_real[train], y_Train_cat[train], validation_data = (X_Train_real[test], y_Train_cat[test]), batch_size = 16, epochs = epoch_num, verbose = 2)
            

            model.save(f'models/{MODEL_NAME}.h5')

            

            for element in range(0,np.size(history.history['val_acc'])):
                validation_accuracies = np.append(validation_accuracies, history.history['val_acc'][element])
                training_accuracies = np.append(training_accuracies, history.history['acc'][element])
                training_losses = np.append(training_losses, history.history['loss'][element])
                validation_losses = np.append(validation_losses, history.history['val_loss'][element])

       

        str_training_accuracies = f'{results_dir}/train_accu_{str_param}.csv'
        str_validation_accuracies = f'{results_dir}/valid_accu_{str_param}.csv'
        str_training_losses = f'{results_dir}/train_loss_{str_param}.csv'
        str_validation_losses = f'{results_dir}/valid_loss_{str_param}.csv'
        #save results
        np.savetxt(str_training_accuracies, training_accuracies)
        np.savetxt(str_validation_accuracies, validation_accuracies)
        np.savetxt(str_training_losses, training_losses)
        np.savetxt(str_validation_losses, validation_losses)

        

        counter = counter + 1
