__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib as edf
import os
# our functions to test and load data
import get_4_cp as get
import data_tester as tst
# tensorflow part
#from tensorflow.keras import utils as np_utils
#from tensorflow.keras.callbacks import ModelCheckpoint
# EEGNet models
#import models as models
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Set data parameters
PATH = "../files/"

# Load only consistent data
X_Train, y_Train = get.get_data(PATH)

c_0 = 0
c_1 = 0 
c_2 = 0 
c_3 = 0 

for i in y_Train:
    if(i == 0):
        c_0 += 1
    elif(i == 1):
        c_1 += 1
    elif(i == 2):
        c_2 += 1
    elif(i == 3):
        c_3 += 1
    else:
        print('ERROR')
        
print(f'count_0:{c_0}, count_1:{c_1}, count_2:{c_2}, count_3:{c_3}')

'''
sub = 1
ru = 1


base_file_name = 'S{:03d}R{:02d}.edf'
base_subject_directory = 'S{:03d}'

filename = base_file_name.format(sub,ru)
directory = base_subject_directory.format(sub)
file_name = os.path.join(PATH,directory,filename)

f = edf.EdfReader(file_name)
# Signal Parameters - measurement frequency
freq = f.getSampleFrequency(0)
# Number of eeg channels = number of signals in file
#n = f.signals_in_file
n = f.signals_in_file
sigbufs = np.zeros((n, f.getNSamples()[0]))
# Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
for i in np.arange(n):
    # Save the read data (vectorwise) in a matrix
    # Fill the matrix with all datapoints from each channel
    sigbufs[i, :] = f.readSignal(i)
# Get Label information
annotations = f.readAnnotations()

print(f'{filename}: {f.file_duration} : {f.getNSamples()[0]}')

# Get the specific label information
labels = annotations[2]
points = freq*annotations[0]


# close the file
f.close()


'''



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

"""