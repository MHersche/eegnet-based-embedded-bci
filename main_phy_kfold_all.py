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
results_dir=f'{current_time.year}-{current_time.month}-{current_time.day}--{current_time.hour}-{current_time.minute}--lr'
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

alphas = [10**i for i in range(-4,-3)]
results = np.zeros(len(alphas))

# create a 2D array for fold creation. # 640 is here the sample size.
x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], 64*SAMPLE_SIZE))


for i in range(len(alphas)):

    # counter for the csv files
    counter = 0
    
    for train, test in kf.split(x_train_aux, y_Train):

        # creating the model every time?
        model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=0.2, kernLength=128, numFilters=8, dropoutType='Dropout')
        
        # compile the model and set the optimizers - Find optimal learning rate between 10e-3 and 10e0
        adam_alpha = Adam(lr=alphas[i]) # originally: optimizer='adam'
        model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

        # count number of parameters in the model
        #numParams    = model.count_params()

        # set a valid path for your system to record model checkpoints
        # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                        # save_best_only=True)

        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        # class_weights = {1:1, 2:1, 3:1, 4:1} # start from 0 or 1 ??

        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], validation_data = (X_Train_real[test], y_Train_cat[test]), batch_size = 16, epochs = 500, verbose = 2)
        
        # too see what is inside can be maybe commented
        print(history.history.keys())

        # assuming you have a csv_files directory
        # another way to have the same functionality but uses more memory
        '''
        import pandas as pd 
        pd.DataFrame(np_array).to_csv("path/to/file.csv")
        '''
        training_accuracies = f'{results_dir}/train_lr[{alphas[i]}]_dr[0.2]_split[{counter}].csv'
        validation_accuracies = f'{results_dir}/valid_lr[{alphas[i]}]_dr[0.2]_split[{counter}].csv'
        training_losses = f'{results_dir}/train_lr_loss[{alphas[i]}]_dr[0.2]_split[{counter}].csv'
        validation_losses = f'{results_dir}/valid_lr_loss[{alphas[i]}]_dr[0.2]_split[{counter}].csv'
        '''
        name_for_train_acc = "csv_files/train_param_alpha" + str(i) + "split" + str(counter) + ".csv"
        name_for_val_acc = "csv_files/test_param_alpha" + str(i) + "split" + str(counter) + ".csv"
        '''
        np.savetxt(training_accuracies, history.history['acc'])
        np.savetxt(validation_accuracies, history.history['val_acc'])
        np.savetxt(training_losses, history.history['loss'])
        np.savetxt(validation_losses, history.history['val_loss'])

        # can be commented out because history object gives the same result for each epoch.
        '''
        probs       = model.predict(X_Train_real[test])
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == y_Train_cat[test].argmax(axis=-1))
        results[i] += acc
        '''

        counter = counter + 1

# do training again with the best model can also be commented for now
'''
lmbda = np.argmax(results)

model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=alphas[lmbda],
                       dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')
model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics = ['accuracy'])

model.fit(X_Train_real, y_Train_cat, batch_size = 16, epochs = 50, verbose = 2)

# load optimal weights
# model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# make prediction on test set. 
###############################################################################

# COMMENT THIS FOR NOW

probs       = model.predict(X_Test_real)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_Test_cat.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
#names        = ['left hand', 'right hand', 'foot', 'tongue']
#plt.figure(0)
#plot_confusion_matrix(preds, Y_eval.argmax(axis = -1), names, title = 'EEGNet-8,2')
'''
