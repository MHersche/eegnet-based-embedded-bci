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
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
PATH = "../files/"
subjects = [1]
"""[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20
            ,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37
            ,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54
            ,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71
            ,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,89
            ,90,91,93,94,95,96,97,98,99,101,102,103,105,107,108,109]"""

training_runs = [4,6,8,10]
test_runs = [12,14]

# Test data: uncomment and run to reproduce above initializes subjects array
#subjects = [i for i in range(1,110)]
#subjects = tst.test_data(subjects, runs, PATH)

# Load only consistent data
X_Train, y_Train = get.get_data(subjects, training_runs, PATH)

X_Test, y_Test = get.get_data(subjects, test_runs, PATH)

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

# convert labels to one-hot encodings.
y_Train      = np_utils.to_categorical(y_Train)
y_Test       = np_utils.to_categorical(y_Test)

# TODO: implement k-fold cross validation on prepared dataset
# TODO: calculate/determine EEGNet parameters

model = models.EEGNet(nb_classes = 4, Chans=64, Samples=640, regRate=.25,
			   dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')

# compile the model and set the optimizers
model.compile(loss='categorical_crossentropy', optimizer='adam', 
              metrics = ['accuracy'])

# count number of parameters in the model
#numParams    = model.count_params()

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                                save_best_only=True)

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
# class_weights = {1:1, 2:1, 3:1, 4:1} # start from 0 or 1 ??

fittedModel = model.fit(X_Train_real,y_Train, batch_size = 16, epochs = 50, verbose = 2)

# load optimal weights
# model.load_weights('/tmp/checkpoint.h5')

###############################################################################
# make prediction on test set.
###############################################################################

probs       = model.predict(X_Test_real)
preds       = probs.argmax(axis = -1)  
acc         = np.mean(preds == y_Test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
#names        = ['left hand', 'right hand', 'foot', 'tongue']
#plt.figure(0)
#plot_confusion_matrix(preds, Y_eval.argmax(axis = -1), names, title = 'EEGNet-8,2')
