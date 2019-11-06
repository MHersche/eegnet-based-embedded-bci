#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data_edf_4class as get
import data_tester as tst
# tensorflow part
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
# EEGNet models
import models as models
from sklearn.model_selection import StratifiedKFold

# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
PATH = "../files/"
subjects = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
            ,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38
            ,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
            ,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
            ,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,89
            ,90,91,93,94,95,96,97,98,99,101,102,103,105,107,108,109]

subjects2 = [1,2,3,4,5,6,7,8,9,10]

training_runs = [4,6,8,10]
test_runs = [12,14]

# Test data: uncomment and run to reproduce above initializes subjects array
#subjects = [i for i in range(1,110)]
#subjects = tst.test_data(subjects, runs, PATH)

# Load only consistent data
X_Train, y_Train = get.get_data(subjects2, training_runs, PATH)

X_Test, y_Test = get.get_data(subjects2, test_runs, PATH)

"""
# Test shuffling - seems to work - delete after making sure
X_Train_cp = X_Train.copy()
y_Train_cp = y_Train.copy()
X_Test_cp = X_Test.copy()
y_Test_cp = y_Test.copy()
tr = [i for i in range(1,54)]
tr_cp = [i for i in range(1,54)]
np.random.seed(42)
np.random.shuffle(tr)
"""

# Suffle data reproducably
np.random.seed(42)
np.random.shuffle(X_Train)
np.random.seed(42)
np.random.shuffle(y_Train)
np.random.seed(42)
np.random.shuffle(X_Test)
np.random.seed(42)
np.random.shuffle(y_Test)

X_Train_real = (np.expand_dims(X_Train, axis=1))
X_Test_real  = (np.expand_dims(X_Test, axis=1))

# use sample size
SAMPLE_SIZE = np.shape(X_Train_real)[3]

# convert labels to one-hot encodings.
y_Train_cat      = np_utils.to_categorical(y_Train)
y_Test_cat       = np_utils.to_categorical(y_Test)

# TODO: implement k-fold cross validation on prepared dataset
# TODO: calculate/determine EEGNet parameters

# using 5 folds
kf = StratifiedKFold(n_splits = 5)

alphas = [10**i for i in range(-3,4)]
results = np.zeros(len(alphas))

# create a 2D array for fold creation. # 640 is here the sample size.
x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], 64*SAMPLE_SIZE))


for i in range(len(alphas)):

    # counter for the csv files
    counter = 0
    
    model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=alphas[i],
                dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')
    
    for train, test in kf.split(x_train_aux, y_Train):
        
        # compile the model and set the optimizers
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                        metrics = ['accuracy'])

        # count number of parameters in the model
        #numParams    = model.count_params()

        # set a valid path for your system to record model checkpoints
        # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                        # save_best_only=True)

        # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
        # the weights all to be 1
        # class_weights = {1:1, 2:1, 3:1, 4:1} # start from 0 or 1 ??

        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], validation_data = (X_Train_real[test], y_Train_cat[test]), batch_size = 16, epochs = 5, verbose = 2)
        
        # too see what is inside can be maybe commented
        print(history.history.keys())

        # assuming you have a csv_files directory
        # another way to have the same functionality but uses more memory
        '''
        import pandas as pd 
        pd.DataFrame(np_array).to_csv("path/to/file.csv")
        '''

        name_for_train_acc = "csv_files/train_param_alpha" + str(i) + "split" + str(counter) + ".csv"
        name_for_val_acc = "csv_files/test_param_alpha" + str(i) + "split" + str(counter) + ".csv"
        np.savetxt(name_for_train_acc, history.history['accuracy'])
        np.savetxt(name_for_val_acc, history.history['val_accuracy'])

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
