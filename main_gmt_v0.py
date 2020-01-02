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
import matplotlib.pyplot as plt
# tools for plotting confusion matrices
#from matplotlib import pyplot as plt
#from conf_matrix import plot_confusion_matrix


#################################################
#
# Learning Rate Constant Scheduling
#
#################################################

def step_decay(epoch):
    if(epoch < 20):
        lr = 0.01
    elif(epoch < 50):
        lr = 0.001
    else:
        lr = 0.0001
    return lr
lrate = LearningRateScheduler(step_decay)


# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'gmt_v0'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# specify number of classses for input data
num_classes_list = [2,3,4]
n_epochs = 100
for num_classes in num_classes_list:
    # Load data
    X_Train, y_Train = get.get_data(PATH, n_classes=num_classes)
    # Expand dimensions to match expected EEGNet input
    X_Train_real = (np.expand_dims(X_Train, axis=1))
    # use sample size
    SAMPLE_SIZE = np.shape(X_Train_real)[3]
    # convert labels to one-hot encodings.
    y_Train_cat = np_utils.to_categorical(y_Train)

    # using 5 folds
    num_splits = 5
    kf = KFold(n_splits = num_splits)
    n_epochs = 100

    split_ctr = 0
    for train, test in kf.split(X_Train_real, y_Train):
        train_accu = np.array([])
        valid_accu = np.array([])
        train_loss = np.array([])
        valid_loss = np.array([])
        epoch_number = 0    
        model = models.EEGNet(nb_classes = num_classes, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=0.2, kernLength=128, numFilters=8, dropoutType='Dropout')
        print(f'Split = {split_ctr}')
        # Set Learning Rate
        adam_alpha = Adam(lr=(0.0001))
        model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
        np.random.seed(42*(split_ctr+1))
        np.random.shuffle(train)
        # creating a history object
        history = model.fit(X_Train_real[train], y_Train_cat[train], 
                validation_data=(X_Train_real[test], y_Train_cat[test]),
                batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)
        train_accu = np.append(train_accu, history.history['acc'])
        valid_accu = np.append(valid_accu, history.history['val_acc'])
        train_loss = np.append(train_loss, history.history['loss'])
        valid_loss = np.append(valid_loss, history.history['val_loss'])

        # Save metrics
        train_accu_str = f'{results_dir}/stats/train_accu_v1_class_{num_classes}_split_{split_ctr}.csv'
        valid_accu_str = f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_split_{split_ctr}.csv'
        train_loss_str = f'{results_dir}/stats/train_loss_v1_class_{num_classes}_split_{split_ctr}.csv'
        valid_loss_str = f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_split_{split_ctr}.csv'
         
        np.savetxt(train_accu_str, train_accu)
        np.savetxt(valid_accu_str, valid_accu)
        np.savetxt(train_loss_str, train_loss)
        np.savetxt(valid_loss_str, valid_loss)

        #Save model
        #print('Saving model...')
        model.save(f'{results_dir}/model/global_class_{num_classes}_split_{split_ctr}_v1.h5')

        #Clear Models
        K.clear_session()
        split_ctr = split_ctr + 1

for num_classes in num_classes_list:
    # Once all CV folds are done, calculate averages, plot, and save
    train_accu = np.zeros(n_epochs)
    valid_accu = np.zeros(n_epochs)
    train_loss = np.zeros(n_epochs)
    valid_loss = np.zeros(n_epochs)
    for split_ctr in range(0,num_splits):
        train_accu_step = np.loadtxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_split_{split_ctr}.csv')
        valid_accu_step = np.loadtxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_split_{split_ctr}.csv')
        train_loss_step = np.loadtxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_split_{split_ctr}.csv')
        valid_loss_step = np.loadtxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_split_{split_ctr}.csv')
        
        train_accu += train_accu_step
        valid_accu += valid_accu_step
        train_loss += train_loss_step
        valid_loss += valid_loss_step

        # Plot Accuracy 
        plt.ylim(0,1)
        plt.plot(train_accu_step, label='Training')
        plt.plot(valid_accu_step, label='Validation')
        plt.title(f'{num_classes}-Class Acc.: LR: 20-30-50, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/accu_{num_classes}_c_global_{split_ctr}.pdf')
        plt.clf()
        # Plot Loss
        plt.plot(train_loss_step, label='Training')
        plt.plot(valid_loss_step, label='Validation')
        plt.title(f'{num_classes}-Class Loss: LR: 20-30-50, DR=0.2')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f'{results_dir}/plots/loss_{num_classes}_c_global_{split_ctr}.pdf')
        plt.clf()
    
    train_accu = train_accu/num_splits
    valid_accu = valid_accu/num_splits
    train_loss = train_loss/num_splits
    valid_loss = valid_loss/num_splits

    np.savetxt(f'{results_dir}/stats/train_accu_v1_class_{num_classes}_avg.csv', train_accu)
    np.savetxt(f'{results_dir}/stats/valid_accu_v1_class_{num_classes}_avg.csv', valid_accu)
    np.savetxt(f'{results_dir}/stats/train_loss_v1_class_{num_classes}_avg.csv', train_loss)
    np.savetxt(f'{results_dir}/stats/valid_loss_v1_class_{num_classes}_avg.csv', valid_loss)

    # Plot Accuracy 
    plt.ylim(0,1)
    plt.plot(train_accu, label='Training')
    plt.plot(valid_accu, label='Validation')
    plt.title(f'{num_classes}-Class Acc.: LR: 20-30-50, DR=0.2')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/accu_avg_{num_classes}_c_global.pdf')
    plt.clf()
    # Plot Loss
    plt.plot(train_loss, label='Training')
    plt.plot(valid_loss, label='Validation')
    plt.title(f'{num_classes}-Class Loss: LR: 20-30-50, DR=0.2')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{results_dir}/plots/loss_avg_{num_classes}_c_global.pdf')
    plt.clf()


