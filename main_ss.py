# Copyright (c) 2020 ETH Zurich, Xiaying Wang, Michael Hersche, Batuhan Toemekce, Burak Kaya, Michele Magno, Luca Benini
#!/usr/bin/env python3

#################################################
# 5 Global models have alredy been trained
# now, these models are used, and further
# (subject-specific) retrained.
# 
# Finally, results within, and across
# subjects are averaged.
#
#################################################


import numpy as np
import os, io, sys
import pdb

from tensorflow.keras import utils as np_utils
from keras.optimizers import Adam
from keras.models import load_model
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import get_data as get

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


#################################################
#
# Remove excluded subjects from subjects list
#
#################################################
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects


#################################################
#
# Learning Rate Constant Scheduling for subject-
# specific transfer learning
#
#################################################
# def step_decay(epoch):
#     if(epoch < 2):
#         lr = 0.01
#     elif(epoch < 5):
#         lr = 0.001
#     else:
#         lr = 0.0001
#     return lr
# lrate = LearningRateScheduler(step_decay)

#################################################
#
# Save results
#
#################################################
def save_results(first_eval,tr_hist,num_classes,sub,split,n_ds,n_ch,T):

    # Save metrics  
    results = np.zeros((4,1+len(tr_hist.history['acc'])))
    # validation results w/o retraining
    results[0,0] = np.nan
    results[1,0] = first_eval[1]
    results[2,0] = np.nan
    results[3,0] = first_eval[0]
    # retraing results
    results[0,1:] = tr_hist.history['acc']
    results[1,1:] = tr_hist.history['val_acc']
    results[2,1:] = tr_hist.history['loss']
    results[3,1:] = tr_hist.history['val_loss']

    sub_str = '{0:03d}'.format(sub)
    results_str = f'{results_dir}/stats/ss_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_sub{sub_str}_split_{split}.csv'
    np.savetxt(results_str, np.transpose(results))
    return results[0:2,-1]

##############################################
# CHANGE EXPERIMENT NAME FOR DIFFERENT TESTS!!
ss_experiment = 'your-ss-experiment'
global_experiment = 'your-global-experiment'
##############################################
datapath = "/usr/scratch/xavier/herschmi/EEG_data/physionet/"

global_model_path = f'results/{global_experiment}/model/'
# Make necessary directories for files
results_dir=f'results/{ss_experiment}'
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# Specify number of classses for input data
num_classes_list = [4]
# Exclude subjects whose data we do not use
subjects = exclude_subjects()
n_subjects = len(subjects)
n_ds = 1
T = 3
n_ch = 64
verbose = 0 # verbosity for data loader and keras: 0 minimum, 

# retraining parameters
n_epochs = 5
lr = 1e-3

for num_classes in num_classes_list:
    # using 5 folds
    num_splits = 5
    kf_global = KFold(n_splits = num_splits)
    
    split_ctr = 0

    acc = np.zeros((n_subjects,4,2))

    # run over 5 global folds 
    for _, test_sub_global in kf_global.split(subjects):
        for sub_idx in test_sub_global:
            
            subject = subjects[sub_idx]
            X_sub, y_sub = get.get_data(datapath, n_classes=num_classes, subjects_list=[subject])
            X_sub = np.expand_dims(X_sub,  axis=-1)
            y_sub_cat = np_utils.to_categorical(y_sub)
            n_samples = np.shape(X_sub)[2]
            # split data while balancing classes 
            kf_subject = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            sub_split_ctr = 0

            for train_sub, test_sub in kf_subject.split(X_sub, y_sub):
                
                # load global model
                model = load_model(global_model_path+f'global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.h5')
                first_eval = model.evaluate(X_sub[test_sub], y_sub_cat[test_sub], batch_size=16, verbose = verbose) 
                
                adam_alpha = Adam(lr=lr)
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                # creating a history object
                history = model.fit(X_sub[train_sub], y_sub_cat[train_sub], 
                        validation_data=(X_sub[test_sub], y_sub_cat[test_sub]),
                        batch_size = 16, epochs = n_epochs,  verbose = verbose) # callbacks=[lrate]
                
                # save results
                acc[sub_idx,sub_split_ctr]=save_results(first_eval,history,num_classes,subject,sub_split_ctr,n_ds,n_ch,T)

                K.clear_session()
                sub_split_ctr = sub_split_ctr + 1
           
            print("S{:d}\t{:.4f}\t{:.4f}".format(subject,acc[sub_idx,:,0].mean(),acc[sub_idx,:,1].mean()))


        split_ctr = split_ctr + 1


    print("AVG\t{:.4f}\t{:.4f}".format(acc[:,:,0].mean(),acc[:,:,1].mean()))

