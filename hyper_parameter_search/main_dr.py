#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_4class as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
# EEGNet models
import models as models
from sklearn.model_selection import StratifiedKFold

# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir  = f'{current_time.year}-{current_time.month}-{current_time.day}--{current_time.hour}-{current_time.minute}--dr'
os.mkdir (results_dir)
        

# Load data
X_Train, y_Train = get.get_data(PATH)
# Expand dimensions to match expected EEGNet input
X_Train_real = (np.expand_dims(X_Train, axis=1))

# use sample size
SAMPLE_SIZE = np.shape(X_Train_real)[3]

# convert labels to one-hot encodings.
y_Train_cat      = np_utils.to_categorical(y_Train)

# using 5 folds
kf = StratifiedKFold(n_splits = 5)
# After hyperparameter search: we chose 0.1 for the dropout rate
alphas = [0.1*i for i in range(1,6)]
results = np.zeros(len(alphas))

# create a 2D array for fold creation. # 640 is here the sample size.
x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], 64*SAMPLE_SIZE))


for i in range(len(alphas)):

    # counter for the csv files
    counter = 0
    
    for train, test in kf.split(x_train_aux, y_Train):

        # creating the model every time?
        model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=alphas[i], kernLength=128, numFilters=8, dropoutType='Dropout')
        
        # compile the model and set the optimizers - Find optimal learning rate between 10e-3 and 10e0
        adam_alpha = Adam(lr=0.0001) # originally: optimizer='adam'
        model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], validation_data = (X_Train_real[test], y_Train_cat[test]), batch_size = 16, epochs = 500, verbose = 2)
        
        # to see what is inside can be maybe commented
        print(history.history.keys())
        dr_str = "{0:.1f}".format(alphas[i])
        training_accuracies = f'{results_dir}/train_lr[0.0001]_dr[{dr_str}]_split[{counter}].csv'
        validation_accuracies = f'{results_dir}/valid_lr[0.0001]_dr[{dr_str}]_split[{counter}].csv'
        training_losses = f'{results_dir}/train_lr_loss[0.0001]_dr[{dr_str}]_split[{counter}].csv'
        validation_losses = f'{results_dir}/valid_lr_loss[0.0001]_dr[{dr_str}]_split[{counter}].csv'
        #save results
        np.savetxt(training_accuracies, history.history['acc'])
        np.savetxt(validation_accuracies, history.history['val_acc'])
        np.savetxt(training_losses, history.history['loss'])
        np.savetxt(validation_losses, history.history['val_loss'])
        
        counter = counter + 1
