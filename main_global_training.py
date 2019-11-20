#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_4class_testdata as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
# EEGNet models
import models as models
from sklearn.model_selection import StratifiedKFold

# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'{current_time.year}-{current_time.month}-{current_time.day}--{current_time.hour}-{current_time.minute}--global_model_training_data'
os.mkdir (results_dir)
        

# Load data
X_Train, y_Train = get.get_data(PATH)
X_Test, y_Test = get.get_data(PATH, train = False)
# Expand dimensions to match expected EEGNet input
X_Train_real = (np.expand_dims(X_Train, axis=1))
X_Test_real = (np.expand_dims(X_Test, axis=1))


# convert labels to one-hot encodings.
y_Train_cat      = np_utils.to_categorical(y_Train)
y_Test_cat       = np_utils.to_categorical(y_Test)

SAMPLE_SIZE = np.shape(X_Train_real)[3]

# creating the model every time?
model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')

# compile the model and set the optimizers - Find optimal learning rate between 10e-3 and 10e0
adam_alpha = Adam(lr=0.0001) # originally: optimizer='adam'
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
history = model.fit(X_Train_real, y_Train_cat, validation_data = (X_Test_real, y_Test_cat), batch_size = 16, epochs = 400, verbose = 2)

# too see what is inside can be maybe commented
print(history.history.keys())

# assuming you have a csv_files directory
# another way to have the same functionality but uses more memory

training_accuracies = f'{results_dir}/train_global.csv'
test_accuracies = f'{results_dir}/test_global.csv'
training_losses = f'{results_dir}/train_loss_global.csv'
test_losses = f'{results_dir}/test_loss_lr_global.csv'

np.savetxt(training_accuracies, history.history['acc'])
np.savetxt(test_accuracies, history.history['val_acc'])
np.savetxt(training_losses, history.history['loss'])
np.savetxt(test_losses, history.history['val_loss'])

print("Saving model...")
model.save('/scratch/gra19h3/global_model_training.h5')


###############################################################################
# make prediction on test set. 
###############################################################################

# COMMENT THIS FOR NOW

probs       = model.predict(X_Test_real)

np.savetxt(f'{results_dir}/probs_global.csv',probs)

preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_Test_cat.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
#names        = ['left hand', 'right hand', 'foot', 'tongue']
#plt.figure(0)
#plot_confusion_matrix(preds, Y_eval.argmax(axis = -1), names, title = 'EEGNet-8,2')
