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
PATH_model = 'global_model_training.h5'

current_time = datetime.now()
results_dir=f'{current_time.year:04}_{current_time.month:02}_{current_time.day:02}_{current_time.hour:02}_{current_time.minute:02}__subject_whole_50_epochs'
os.mkdir (results_dir)

excluded_subjects = [88,92,100,104,106]

for subject_id in range(1,110):
	# Do not do subject-specific training on subjects whose data is irregular
	if subject_id in excluded_subjects:
		print(f'Skipping excluded subject: {subject_id} ...')		
		continue

	# Load data
	X_Train, y_Train = get.get_data(PATH, subjects_list=[subject_id])
	X_Test, y_Test   = get.get_data(PATH, subjects_list=[subject_id] ,train = False)
	# Expand dimensions to match expected EEGNet input
	X_Train_real     = (np.expand_dims(X_Train, axis=1))
	X_Test_real      = (np.expand_dims(X_Test, axis=1))

	# convert labels to one-hot encodings.
	y_Train_cat      = np_utils.to_categorical(y_Train)
	y_Test_cat       = np_utils.to_categorical(y_Test)

	SAMPLE_SIZE = np.shape(X_Train_real)[3]

	# load globally trained model
	model = load_model(PATH_model)
	
	# First evaluate once with global model before further subject specific training
	first_eval = model.evaluate(X_Test_real, y_Test_cat, batch_size = 16)
	
	# create history object to store epoch results
	history = model.fit(X_Train_real, y_Train_cat, validation_data = (X_Test_real, y_Test_cat), batch_size = 16, epochs = 50, verbose = 2)

	# save results
	subject_str = '{0:03d}'.format(subject_id)
	first_eval_both     = f'{results_dir}/{subject_str}_first_both_glocal.csv'
	training_accuracies = f'{results_dir}/{subject_str}_train_accu_global.csv'
	test_accuracies     = f'{results_dir}/{subject_str}_test-_accu_global.csv'
	training_losses     = f'{results_dir}/{subject_str}_train_loss_global.csv'
	test_losses         = f'{results_dir}/{subject_str}_test-_loss_global.csv'
	
	np.savetxt(first_eval_both, first_eval)
	
	np.savetxt(training_accuracies, history.history['acc'])
	np.savetxt(test_accuracies, history.history['val_acc'])
	np.savetxt(training_losses, history.history['loss'])
	np.savetxt(test_losses, history.history['val_loss'])

	###############################################################################
	# make prediction on test set. 
	###############################################################################

	# COMMENT THIS FOR NOW

	probs       = model.predict(X_Test_real)
	
	np.savetxt(f'{results_dir}/{subject_str}_probs_global.csv',probs)

	preds       = probs.argmax(axis = -1)  
	acc         = np.mean(preds == y_Test_cat.argmax(axis=-1))
	print("Classification accuracy: %f " % (acc))

	# plot the confusion matrices for both classifiers
	#names        = ['left hand', 'right hand', 'foot', 'tongue']
	#plt.figure(0)
	#plot_confusion_matrix(preds, Y_eval.argmax(axis = -1), names, title = 'EEGNet-8,2')
