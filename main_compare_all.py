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
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
# EEGNet models
import models as models
from sklearn.model_selection import KFold

# Set data parameters
PATH = "../files/"

current_time = datetime.now()
results_dir=f'{current_time.month:02}_{current_time.day:02}_{current_time.hour:02}_{current_time.minute:02}_comparison_paper_m_class_patience200'
os.mkdir (results_dir)

# specify number of classses for input data
num_classes_list = [2,3,4] # [2,3,4]       

for num_classes in num_classes_list:
    
    # Load data
    X_Train, y_Train = get.get_data(PATH, n_classes=num_classes)

    # Expand dimensions to match expected EEGNet input
    X_Train_real = (np.expand_dims(X_Train, axis=1))

    # use sample size
    SAMPLE_SIZE = np.shape(X_Train_real)[3]

    # convert labels to one-hot encodings.
    y_Train_cat      = np_utils.to_categorical(y_Train)

    # using 5 folds
    kf = KFold(n_splits = 5)

    # create a 2D array for fold creation. # 640 is here the sample size.
    x_train_aux = np.reshape(X_Train_real, (np.shape(X_Train_real)[0], 64*SAMPLE_SIZE))
    
    # counter for the csv files
    counter = 0

    for train, test in kf.split(x_train_aux, y_Train):

        np.random.seed(100)
        np.random.shuffle(train)
        
        model = models.EEGNet(nb_classes = 4, Chans=64, Samples=SAMPLE_SIZE, regRate=0.25,
                        dropoutRate=0.1, kernLength=128, numFilters=8, dropoutType='Dropout')
        
        # compile the model and set the optimizers
        adam_alpha = Adam(lr=0.0001) # originally: optimizer='adam'
        model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])

        # set a valid path for your system to record model checkpoints
        checkpointer = ModelCheckpoint(filepath=f'./tmp/checkpoint_4class_{counter}.h5', verbose=1,
                                        save_best_only=True, monitor="val_acc")
        early_stopping = EarlyStopping(monitor='val_acc', patience=200, verbose=0, mode='max')

        # train the model
        history = model.fit(X_Train_real[train], y_Train_cat[train], validation_data = (X_Train_real[test], y_Train_cat[test]), batch_size = 16, epochs = 1000, verbose = 2, callbacks = [checkpointer, early_stopping])
        
        print(history.history.keys())

        train_accu_str = f'{results_dir}/train_accu_lr_1e-4_dr_01_split_{counter}.csv'
        valid_accu_str = f'{results_dir}/valid_accu_lr_1e-4_dr_01_split_{counter}.csv'
        train_loss_str = f'{results_dir}/train_loss_lr_1e-4_dr_01_split_{counter}.csv'
        valid_loss_str = f'{results_dir}/valid_loss_lr_1e-4_dr_01_split_{counter}.csv'

        np.savetxt(train_accu_str, history.history['acc'])
        np.savetxt(valid_accu_str, history.history['val_acc'])
        np.savetxt(train_loss_str, history.history['loss'])
        np.savetxt(valid_loss_str, history.history['val_loss'])

        model       = load_models(f'./tmp/checkpoint_4class_{counter}.h5')
        probs       = model.predict(X_Train_real[test])
        np.savetxt(f'{results_dir}/probs_split_4class_{counter}',probs)
        preds       = probs.argmax(axis = -1)  
        acc         = np.mean(preds == y_Train_cat[test].argmax(axis=-1))
        print("Classification accuracy: %f " % (acc))

        counter = counter + 1