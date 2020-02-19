#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
# EEGNet models
import models as models
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from eeg_reduction import eeg_reduction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "3"



#################################################
#
# Learning Rate Constant Scheduling
#
#################################################

def step_decay(epoch):
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)


# Set data parameters
PATH = "/usr/scratch/xavier/herschmi/EEG_data/physionet/"

current_time = datetime.now()
results_dir=f'../results/M4M7_models'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# specify number of classses for input data
num_classes_list = [4]
n_epochs = 100
num_splits = 5

# data settings  
n_ds = 3 # downsamlping factor [1,2,3]
n_ch_vec = [38] # number of channels [19,27,38,64]
T_vec = [2] # duration to classify

# model settings 
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))


for num_classes in num_classes_list:
    for n_ch in n_ch_vec:
        for T in T_vec:

            # Load data
            #X_Train, y_Train = get.get_data(PATH, n_classes=num_classes)

            
            #np.savez(PATH+f'{num_classes}class',X_Train = X_Train, y_Train = y_Train)
            npzfile = np.load(PATH+f'{num_classes}class.npz')
            X_Train, y_Train = npzfile['X_Train'], npzfile['y_Train']

            X_Train = eeg_reduction(X_Train,n_ds = n_ds, n_ch = n_ch, T = T)

            # Expand dimensions to match expected EEGNet input
            X_Train_real = (np.expand_dims(X_Train, axis=1))
            # use sample size
            SAMPLE_SIZE = np.shape(X_Train_real)[3]
            # convert labels to one-hot encodings.
            y_Train_cat = np_utils.to_categorical(y_Train)

            # using 5 folds
           
            kf = KFold(n_splits = num_splits)

            split_ctr = 0
            for train, test in kf.split(X_Train_real, y_Train):
                train_accu = np.array([])
                valid_accu = np.array([])
                train_loss = np.array([])
                valid_loss = np.array([])
                epoch_number = 0    
                model = models.EEGNet(nb_classes = num_classes, Chans=n_ch, Samples=SAMPLE_SIZE, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, dropoutType='Dropout')


                print(model.summary())

                print(f'Split = {split_ctr}')
                # Set Learning Rate
                adam_alpha = Adam(lr=(0.0001))
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                np.random.seed(42*(split_ctr+1))
                np.random.shuffle(train)
                # creating a history object
                history = model.fit(X_Train_real[train], y_Train_cat[train], 
                        validation_data=(X_Train_real[test], y_Train_cat[test]),
                        batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)
                train_accu = np.append(train_accu, history.history['acc'])
                valid_accu = np.append(valid_accu, history.history['val_acc'])
                train_loss = np.append(train_loss, history.history['loss'])
                valid_loss = np.append(valid_loss, history.history['val_loss'])

                # Save metrics
                train_accu_str = f'{results_dir}/stats/train_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                valid_accu_str = f'{results_dir}/stats/valid_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                train_loss_str = f'{results_dir}/stats/train_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                valid_loss_str = f'{results_dir}/stats/valid_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                 
                np.savetxt(train_accu_str, train_accu)
                np.savetxt(valid_accu_str, valid_accu)
                np.savetxt(train_loss_str, train_loss)
                np.savetxt(valid_loss_str, valid_loss)

                #Save model
                #print('Saving model...')
                model.save(f'{results_dir}/model/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}_v1.h5')

                #Clear Models
                K.clear_session()
                split_ctr = split_ctr + 1

for num_classes in num_classes_list:
     for n_ch in n_ch_vec:
        for T in T_vec:
            # Once all CV folds are done, calculate averages, plot, and save
            train_accu = np.zeros(n_epochs)
            valid_accu = np.zeros(n_epochs)
            train_loss = np.zeros(n_epochs)
            valid_loss = np.zeros(n_epochs)
            for split_ctr in range(0,num_splits):
                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv')
                
                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step

                    
            train_accu = train_accu/num_splits
            valid_accu = valid_accu/num_splits
            train_loss = train_loss/num_splits
            valid_loss = valid_loss/num_splits

            print("{:}-Fold Validation Accuracy {:.4f}".format(num_splits, valid_accu[-1]))

            np.savetxt(f'{results_dir}/stats/train_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/valid_accu_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/train_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/valid_loss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_avg.csv', valid_loss)

           


