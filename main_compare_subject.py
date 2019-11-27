#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

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
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
# EEGNet models
import models as models
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold

# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'{current_time.month:02}_{current_time.day:02}_{current_time.hour:02}_{current_time.minute:02}_comparison_paper_m_class_subjects'
os.mkdir (results_dir)

# specify number of classses for input data
num_classes_list = [2,3,4] # [2,3,4]     
subjects = np.arange(1,110)
excluded_subjects = [88,92,100,104]
np.delete(subjects,excluded_subjects)
kf_global = KFold(n_splits=5)


for num_classes in num_classes_list:
    counter_global = 0
    for train_global, test_global in kf_global.split(subjects):
        subject_list = subjects[test_global]
        for subject in subject_list:
            X_subject, Y_subject = get.get_data(PATH,n_classes= num_classes, subjects_list=[subject])
            X_subject = np.expand_dims(X_subject,axis=1)
            Y_subject_onehot = np_utils.to_categorical(Y_subject)
            SAMPLE_SIZE = np.shape(X_subject)[3]
            kf_subject = StratifiedKFold(n_splits=4, shuffle=True, random_state=100)
            x_subject_step = np.reshape(X_subject, (np.shape(X_subject)[0], 64*SAMPLE_SIZE))

            counter_subject_split = 0
            for train_subject, test_subject in kf_subject.split(x_subject_step, Y_subject):

                model = load_model(f'./tmp/checkpoint_{num_classes}class_{counter_global}.h5')

                history = model.fit(X_subject[train_subject], Y_subject_onehot[train_subject], validation_data = (X_subject[test_subject], Y_subject_onehot[test_subject]), batch_size = 16, epochs = 10, verbose = 2)

                subject_str = '{0:03d}'.format(subject)
                train_accu_str = f'{results_dir}/train_accu_lr_1e-4_dr_01_split__{num_classes}class_{counter_subject_split}_subject_{subject_str}.csv'
                valid_accu_str = f'{results_dir}/valid_accu_lr_1e-4_dr_01_split__{num_classes}class_{counter_subject_split}_subject_{subject_str}.csv'
                train_loss_str = f'{results_dir}/train_loss_lr_1e-4_dr_01_split__{num_classes}class_{counter_subject_split}_subject_{subject_str}.csv'
                valid_loss_str = f'{results_dir}/valid_loss_lr_1e-4_dr_01_split__{num_classes}class_{counter_subject_split}_subject_{subject_str}.csv'

                np.savetxt(train_accu_str, history.history['acc'])
                np.savetxt(valid_accu_str, history.history['val_acc'])
                np.savetxt(train_loss_str, history.history['loss'])
                np.savetxt(valid_loss_str, history.history['val_loss'])

                probs       = model.predict(X_subject[test_subject])
                np.savetxt(f'{results_dir}/probs_split_{num_classes}class_{counter_subject_split}_subject_{subject_str}.csv',probs)
                preds       = probs.argmax(axis = -1)  
                acc         = np.mean(preds == Y_subject_onehot[test_subject].argmax(axis=-1))
                print("Classification accuracy: %f " % (acc))

                counter_subject_split = counter_subject_split + 1
        counter_global = counter_global + 1
