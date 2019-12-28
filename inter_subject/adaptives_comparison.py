#!/usr/bin/env python3
__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import sys
sys.path.append('../')
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
import keras
from keras.callbacks import LearningRateScheduler
#from tensorflow.keras import optimizers
from keras.models import load_model
from keras import backend as K
# EEGNet models
import models as models
from sklearn.model_selection import KFold


# Remove excluded subjects from subjects list
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

RMSprop  =  keras.optimizers.RMSprop()
Adagrad  =  keras.optimizers.Adagrad()
Adadelta =  keras.optimizers.Adadelta() 
Adam     =  keras.optimizers.Adam()
Adamax   =  keras.optimizers.Adamax() 
Nadam    =  keras.optimizers.Nadam()

optimizers = [RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam]
opt_names = ['RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

# Set data parameters
PATH = "../../files/"

results_dir=f'Adaptives'
os.makedirs(f'{results_dir}/stats', exist_ok=True)

# specify number of classses for input data
num_classes = 4       

# Using the first 84 subjects' data
X_Train, y_Train = get.get_data(PATH, n_classes=num_classes, subjects_list=range(1,85))

# Expand dimensions to match expected EEGNet input
X_Train_real = (np.expand_dims(X_Train, axis=1))

# use sample size
SAMPLE_SIZE = np.shape(X_Train_real)[3]

# convert labels to one-hot encodings.
y_Train_cat = np_utils.to_categorical(y_Train)

# using 4 folds
num_splits = 4
kf = KFold(n_splits = num_splits)

n_epochs = 100

for j, optimizer in enumerate(optimizers):
    print(f'Using optimizer:{opt_names[j]}')
    train_accu_step = np.zeros(n_epochs)
    valid_accu_step = np.zeros(n_epochs)
    train_loss_step = np.zeros(n_epochs)
    valid_loss_step = np.zeros(n_epochs)
    split_ctr = 0
    for train, test in kf.split(X_Train_real, y_Train):
        
        np.random.seed(42)
        np.random.shuffle(train)

        print(f'Split = {split_ctr}')
        model = models.EEGNet(nb_classes = num_classes, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=0.2, kernLength=128, numFilters=8, dropoutType='Dropout')
        # Set Learning Rate
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['accuracy'])
        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], 
                validation_data=(X_Train_real[test], y_Train_cat[test]),
                batch_size = 16, epochs = n_epochs, verbose = 2)

        # Save metrics
        train_accu_str = f'{results_dir}/stats/train_accu_{opt_names[j]}_s{split_ctr}.csv'
        valid_accu_str = f'{results_dir}/stats/valid_accu_{opt_names[j]}_s{split_ctr}.csv'
        train_loss_str = f'{results_dir}/stats/train_loss_{opt_names[j]}_s{split_ctr}.csv'
        valid_loss_str = f'{results_dir}/stats/valid_loss_{opt_names[j]}_s{split_ctr}.csv'
         
        np.savetxt(train_accu_str, history.history['acc'])
        np.savetxt(valid_accu_str, history.history['val_acc'])
        np.savetxt(train_loss_str, history.history['loss'])
        np.savetxt(valid_loss_str, history.history['val_loss'])
       
        train_accu_step += history.history['acc']
        valid_accu_step += history.history['val_acc']
        train_loss_step += history.history['loss']
        valid_loss_step += history.history['val_loss']

        #Clear Models
        #K.clear_session()
        split_ctr = split_ctr + 1
    
    train_accu_str = f'{results_dir}/stats/train_accu_{opt_names[j]}_avg.csv'
    valid_accu_str = f'{results_dir}/stats/valid_accu_{opt_names[j]}_avg.csv'
    train_loss_str = f'{results_dir}/stats/train_loss_{opt_names[j]}_avg.csv'
    valid_loss_str = f'{results_dir}/stats/valid_loss_{opt_names[j]}_avg.csv'

    train_accu = train_accu_step/num_splits
    valid_accu = valid_accu_step/num_splits
    train_loss = train_loss_step/num_splits
    valid_loss = valid_loss_step/num_splits

    np.savetxt(train_accu_str, train_accu)
    np.savetxt(valid_accu_str, valid_accu)
    np.savetxt(train_loss_str, train_loss)
    np.savetxt(valid_loss_str, valid_loss)
