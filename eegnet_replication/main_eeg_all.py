#!/usr/bin/env python3

import models as models
import get_data as get_data

import numpy as np

# tensorflow part
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint

# tools for plotting confusion matrices
from matplotlib import pyplot as plt

from conf_matrix import plot_confusion_matrix

general_acc = 0

for ii in range (1, 10):

	# data part
	X_train , Y_train = get_data.get_data(ii, True, 'dataset/')
	X_eval, Y_eval    = get_data.get_data(ii, False, 'dataset/')

	X_train_real = (np.expand_dims(X_train, axis=1))
	X_eval_real  = (np.expand_dims(X_eval, axis=1))

	X_train_real = X_train_real[:,:,:,375:1500];
	X_eval_real  = X_eval_real[:,:,:,375:1500];


	# convert labels to one-hot encodings.
	Y_train      = np_utils.to_categorical(Y_train-1)
	Y_eval       = np_utils.to_categorical(Y_eval-1)

	# print('X_train shape:')
	# print(X_train_real[0], 'train samples')
	# print(Y_train.astype(np.int)[0], 'labels')

	# EEGNet part
		
	model = models.EEGNet(nb_classes = 4, Chans=22, Samples=1125, regRate=.25,
			   dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')

	# compile the model and set the optimizers
	model.compile(loss='categorical_crossentropy', optimizer='adam', 
        	     	metrics = ['accuracy'])

	# count number of parameters in the model
	numParams    = model.count_params()



	# set a valid path for your system to record model checkpoints
	# checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
	  #                               save_best_only=True)

	# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
	# the weights all to be 1
	# class_weights = {1:1, 2:1, 3:1, 4:1} # start from 0 or 1 ??

	fittedModel = model.fit(X_train_real,Y_train, batch_size = 16, epochs = 500, verbose = 2)

	# load optimal weights
	# model.load_weights('/tmp/checkpoint.h5')

	###############################################################################
	# make prediction on test set.
	###############################################################################

	probs       = model.predict(X_eval_real)
	preds       = probs.argmax(axis = -1)  
	acc         = np.mean(preds == Y_eval.argmax(axis=-1))
	general_acc = general_acc + acc;
	print("Classification accuracy of" , ii," : %f " % (acc))
general_acc = general_acc / 9
print("General accuracy: %f " % (general_acc))

	# plot the confusion matrices for both classifiers
	# names        = ['left hand', 'right hand', 'foot', 'tongue']
	# plt.figure(0)
	# plot_confusion_matrix(preds, Y_eval.argmax(axis = -1), names, title = 'EEGNet-8,2')
