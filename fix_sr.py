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
#
#################################################

# Number of epochs to use with 10^[-3,-4,-5] Learning Rate
epochs = [2,3,5]
lrates = [-3,-4,-5]


# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'subject_specific'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)
num_splits = 5
n_epochs = 10
# specify number of classses for input data
num_classes_list = [2,3,4]
subjects = exclude_subjects()
kf_global = KFold(n_splits=5)
for num_classes in num_classes_list:
    split_ctr = 0
    for train_global, test_global in kf_global.split(subjects):
        for sub_idx in test_global:
            subject = subjects[sub_idx]
            train_accu = np.zeros(n_epochs+1)
            valid_accu = np.zeros(n_epochs+1)
            train_loss = np.zeros(n_epochs+1)
            valid_loss = np.zeros(n_epochs+1)
            sub_str = '{0:03d}'.format(subject)
            for sub_split_ctr in range(0,4):
                # Save metrics
                train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
                valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_fold_{sub_split_ctr}.csv')
           
                train_accu += train_accu_step
                valid_accu += valid_accu_step
                train_loss += train_loss_step
                valid_loss += valid_loss_step
            
            train_accu = train_accu/num_splits
            valid_accu = valid_accu/num_splits
            train_loss = train_loss/num_splits
            valid_loss = valid_loss/num_splits

            np.savetxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_accu)
            np.savetxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_accu)
            np.savetxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', train_loss)
            np.savetxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_subject_{sub_str}_avg.csv', valid_loss)

            # Plot Accuracy 
            plt.plot(train_accu, label='Training')
            plt.plot(valid_accu, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Acc.: LR: 2-3-5, DR=0.2')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/accu_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()
            # Plot Loss
            plt.plot(train_loss, label='Training')
            plt.plot(valid_loss, label='Validation')
            plt.title(f'S:{sub_str} C:{num_classes} Loss: LR: 2-3-5, DR=0.2')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(f'{results_dir}/plots/loss_avg_{num_classes}_c_{sub_str}.pdf')
            plt.clf()

        split_ctr = split_ctr + 1
