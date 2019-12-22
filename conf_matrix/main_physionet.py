#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import models
from get_data_edf import get_data
import numpy as np
from sklearn.model_selection import KFold

# tensorflow part
from tensorflow.keras import utils as np_utils
# from tensorflow.keras.callbacks import ModelCheckpoint

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from conf_matrix import plot_confusion_matrix

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

# data part

subjects = np.arange(1,110)
runs = np.arange(3,15)

X, Y = get_data(subjects,runs,'')

# Seperate test and train data
X_train = 
Y_train = 
X_test = 
Y_test = 


# convert labels to one-hot encodings.
Y = np_utils.to_categorical(Y)

# Do KFold cross validation

kf = KFold(4, shuffle=True, random_state=42)


fold = 0
for train, test in kf.split(X_train):
    fold += 1
    print(f"Fold #{fold}")
          
    X_train_train = X_train[train]
    Y_train_train = Y_train[train]
    X_train_validation = X_train[test]
    Y_train_validation = Y_train[test]

    # EEGNet part
    
    model = models.EEGNet(nb_classes = 4, Chans=64, Samples=656, regRate=.25,
    			   dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')
    
    # compile the model and set the optimizers
    model.compile(loss='categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    # count number of parameters in the model
    numParams    = model.count_params()
    
    # set a valid path for your system to record model checkpoints
    # checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                   # save_best_only=True)
    
    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    # class_weights = {1:1, 2:1, 3:1, 4:1} # start from 0 or 1 ??
    
    fittedModel = model.fit(X_train,Y_train, batch_size = 16, epochs = 3, verbose = 2, shuffle=True, validation_set=0.25)
    
    # load optimal weights
    # model.load_weights('/tmp/checkpoint.h5')
    
    ###############################################################################
    # make prediction on test set.
    ###############################################################################
    
    probs       = model.predict(X_test)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == Y_test.argmax(axis=-1))
    print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
names        = ['rest', 'right hand', 'left hand', 'feet']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
# saving the confusion matrix
plt.savefig("conf_matrix.png")