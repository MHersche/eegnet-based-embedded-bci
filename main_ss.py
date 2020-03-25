# Copyright (c) 2020 ETH Zurich, Xiaying Wang, Michael Hersche, Batuhan Toemekce, Burak Kaya, Michele Magno, Luca Benini
#!/usr/bin/env python3


import numpy as np
import os

# our functions to test and load data
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
# EEGNet models
import models as models
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import get_data as get
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix

# Remove excluded subjects from subjects list
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

#################################################
#
# Learning Rate Sparse Scheduling 
# used on Subject Specific Retraining
#
# 5 Global models have alredy been trained
# now, these models are used, and further
# (subject-specifically) retrained.
# 
# Finally, results within, and across
# subjects are averaged.
#
#################################################

#################################################
#
# Learning Rate Constant Scheduling
#
#################################################

def step_decay(epoch):
    if(epoch < 2):
        lr = 0.01
    elif(epoch < 5):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)

# Set data path
PATH = "../files/"
# Make necessary directories for files
results_dir=f'gmt_v0_ss_v0'
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)
os.makedirs(f'{results_dir}/stats/avg', exist_ok=True)
os.makedirs(f'{results_dir}/plots/avg', exist_ok=True)

# Specify number of classses for input data
num_classes_list = [2,3,4]
# Exclude subjects whose data we do not use
subjects = exclude_subjects()

for num_classes in num_classes_list:
    # using 5 folds
    num_splits = 5
    kf_global = KFold(n_splits = num_splits)
    n_epochs = 10

    split_ctr = 0
    for train_global, test_global in kf_global.split(subjects):
        for sub_idx in test_global:
            subject = subjects[sub_idx]
            X_sub, y_sub = get.get_data(PATH, n_classes=num_classes, subjects_list=[subject])
            X_sub = np.expand_dims(X_sub, axis=1)
            y_sub_cat = np_utils.to_categorical(y_sub)
            SAMPLE_SIZE = np.shape(X_sub)[3]
            kf_subject = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            sub_split_ctr = 0

            for train_sub, test_sub in kf_subject.split(X_sub, y_sub):
                print(f'N_Classes:{num_classes}, Model: {split_ctr} \n Subject: {subject:03d}, Split: {sub_split_ctr}')
                model = load_model(f'gmt_v0/model/global_class_{num_classes}_split_{split_ctr}_v1.h5')
                first_eval = model.evaluate(X_sub[test_sub], y_sub_cat[test_sub], batch_size=16) 
                train_accu = np.array([])
                valid_accu = np.array([])
                train_loss = np.array([])
                valid_loss = np.array([])
                # The first elements of the arrays from evaluation
                train_accu = np.append(train_accu, first_eval[1])
                valid_accu = np.append(valid_accu, first_eval[1])
                train_loss = np.append(train_loss, first_eval[0])
                valid_loss = np.append(valid_loss, first_eval[0])
                adam_alpha = Adam(lr=(0.0001))
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                # creating a history object
                history = model.fit(X_sub[train_sub], y_sub_cat[train_sub], 
                        validation_data=(X_sub[test_sub], y_sub_cat[test_sub]),
                        batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)
                train_accu = np.append(train_accu, history.history['acc'])
                valid_accu = np.append(valid_accu, history.history['val_acc'])
                train_loss = np.append(train_loss, history.history['loss'])
                valid_loss = np.append(valid_loss, history.history['val_loss'])
                sub_str = '{0:03d}'.format(subject)
                # Save metrics
                train_accu_str = f'{results_dir}/stats/train_accu_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv'
                valid_accu_str = f'{results_dir}/stats/valid_accu_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv'
                train_loss_str = f'{results_dir}/stats/train_loss_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv'
                valid_loss_str = f'{results_dir}/stats/valid_loss_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv'
                     
                np.savetxt(train_accu_str, train_accu)
                np.savetxt(valid_accu_str, valid_accu)
                np.savetxt(train_loss_str, train_loss)
                np.savetxt(valid_loss_str, valid_loss)

                K.clear_session()
                sub_split_ctr = sub_split_ctr + 1
        split_ctr = split_ctr + 1

for num_classes in num_classes_list:
    os.makedirs(f'{results_dir}/stats/{num_classes}_class', exist_ok=True)
    os.makedirs(f'{results_dir}/plots/{num_classes}_class', exist_ok=True)
    for subject in subjects:
        train_accu = np.zeros(n_epochs+1)
        valid_accu = np.zeros(n_epochs+1)
        train_loss = np.zeros(n_epochs+1)
        valid_loss = np.zeros(n_epochs+1)
        sub_str = '{0:03d}'.format(subject)
        for sub_split_ctr in range(0,4):
            # Save metrics
            train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv')
            valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv')
            train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv')
            valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v00_c_{num_classes}_sub_{sub_str}_fold_{sub_split_ctr}.csv')
       
            train_accu += train_accu_step
            valid_accu += valid_accu_step
            train_loss += train_loss_step
            valid_loss += valid_loss_step
        
        train_accu = train_accu/4
        valid_accu = valid_accu/4
        train_loss = train_loss/4
        valid_loss = valid_loss/4

        np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v00_c_{num_classes}_sub_{sub_str}_avg.csv', train_accu)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v00_c_{num_classes}_sub_{sub_str}_avg.csv', valid_accu)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v00_c_{num_classes}_sub_{sub_str}_avg.csv', train_loss)
        np.savetxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v00_c_{num_classes}_sub_{sub_str}_avg.csv', valid_loss)
for num_classes in num_classes_list:
    train_accu = np.zeros(n_epochs+1)
    valid_accu = np.zeros(n_epochs+1)
    train_loss = np.zeros(n_epochs+1)
    valid_loss = np.zeros(n_epochs+1)
    for subject in subjects:
        sub_str = '{0:03d}'.format(subject)
        train_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_accu_v00_c_{num_classes}_sub_{sub_str}_avg.csv')
        valid_accu_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_accu_v00_c_{num_classes}_sub_{sub_str}_avg.csv')
        train_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/train_loss_v00_c_{num_classes}_sub_{sub_str}_avg.csv')
        valid_loss_step = np.loadtxt(f'{results_dir}/stats/{num_classes}_class/valid_loss_v00_c_{num_classes}_sub_{sub_str}_avg.csv')
   
        train_accu += train_accu_step
        valid_accu += valid_accu_step
        train_loss += train_loss_step
        valid_loss += valid_loss_step
    
    train_accu = train_accu/len(subjects)
    valid_accu = valid_accu/len(subjects)
    train_loss = train_loss/len(subjects)
    valid_loss = valid_loss/len(subjects)

    np.savetxt(f'{results_dir}/stats/avg/train_accu_v00_c_{num_classes}_ss_avg.csv', train_accu)
    np.savetxt(f'{results_dir}/stats/avg/valid_accu_v00_c_{num_classes}_ss_avg.csv', valid_accu)
    np.savetxt(f'{results_dir}/stats/avg/train_loss_v00_c_{num_classes}_ss_avg.csv', train_loss)
    np.savetxt(f'{results_dir}/stats/avg/valid_loss_v00_c_{num_classes}_ss_avg.csv', valid_loss)

    # Plot Accuracy 
    plt.plot(train_accu, label='Training')
    plt.plot(valid_accu, label='Validation')
    plt.title(f'{num_classes}-Class SS Classifier Acc.')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/avg/accu_avg_{num_classes}_c.pdf')
    plt.clf()
    # Plot Loss
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title(f'{num_classes}-Class SS Classifier Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/avg/loss_avg_{num_classes}_c.pdf')
    plt.clf()
