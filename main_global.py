# Copyright (c) 2020 ETH Zurich, Xiaying Wang, Michael Hersche, Batuhan Toemekce, Burak Kaya, Michele Magno, Luca Benini
#!/usr/bin/env python3

#################################################
#
# Global model training 
#
#################################################


import numpy as np
import os
# 
import get_data as get
from tensorflow.keras import utils as np_utils
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras import backend as K
from sklearn.model_selection import KFold

# EEGNet models
import models as models
# Channel reduction, downsampling, time window
from eeg_reduction import eeg_reduction

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


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

#################################################
#
# Save results
#
#################################################
def save_results(history,num_classes,n_ds,n_ch,T,split_ctr):

    # Save metrics  
    results = np.zeros((4,len(history.history['acc'])))
    results[0] = history.history['acc']
    results[1] = history.history['val_acc']
    results[2] = history.history['loss']
    results[3] = history.history['val_loss']
    results_str = f'{results_dir}/stats/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.csv'
                 
    np.savetxt(results_str, np.transpose(results))
    return results[0:2,-1]



# Directories
datapath = "/usr/scratch/xavier/herschmi/EEG_data/physionet/"
results_dir=f'../results/trash'
#os.makedirs(results_dir, exist_ok=True)
os.makedirs(f'{results_dir}/stats', exist_ok=True)
os.makedirs(f'{results_dir}/model', exist_ok=True)
os.makedirs(f'{results_dir}/plots', exist_ok=True)

# specify number of classses for input data
num_classes_list = [4]
n_epochs = 100
num_splits = 5

# data settings  
n_ds = 1 # downsamlping factor {1,2,3}
n_ch_list = [64] # number of channels {8,19,27,38,64}
T_list = [3] # duration to classify {1,2,3}
acc = np.zeros((num_splits,2))
# model settings 
kernLength = int(np.ceil(128/n_ds))
poolLength = int(np.ceil(8/n_ds))


for num_classes in num_classes_list:
    for n_ch in n_ch_list:
        for T in T_list:

            # Load data
            #X, y = get.get_data(datapath, n_classes=num_classes)

            #np.savez(datapath+f'{num_classes}class',X_Train = X_Train, y_Train = y_Train)
            npzfile = np.load(datapath+f'{num_classes}class.npz')
            X, y = npzfile['X_Train'], npzfile['y_Train']

            # reduce EEG data (downsample, number of channels, time window)
            X = eeg_reduction(X,n_ds = n_ds, n_ch = n_ch, T = T)

            # Expand dimensions to match expected EEGNet input
            X = (np.expand_dims(X, axis=-1))
            # number of temporal sample per trial
            n_samples = np.shape(X)[2]
            # convert labels to one-hot encodings.
            y_cat = np_utils.to_categorical(y)

            # using 5 folds
            kf = KFold(n_splits = num_splits)

            split_ctr = 0
            for train, test in kf.split(X, y):
                
                # init model 
                model = models.EEGNet(nb_classes = num_classes, Chans=n_ch, Samples=n_samples, regRate=0.25,
                                dropoutRate=0.2, kernLength=kernLength, poolLength=poolLength, numFilters=8, 
                                dropoutType='Dropout')
               
                #print(model.summary())

                # Set Learning Rate
                adam_alpha = Adam(lr=(0.0001))
                model.compile(loss='categorical_crossentropy', optimizer=adam_alpha, metrics = ['accuracy'])
                np.random.seed(42*(split_ctr+1))
                np.random.shuffle(train)
                # do training
                history = model.fit(X[train], y_cat[train], 
                        validation_data=(X[test], y_cat[test]),
                        batch_size = 16, epochs = n_epochs, callbacks=[lrate], verbose = 2)

                acc[split_ctr] = save_results(history,num_classes,n_ds,n_ch,T,split_ctr)
                
                print('Fold {:}\t{:.4f}\t{:.4f}'.format(split_ctr,acc[split_ctr,0], acc[split_ctr,1]))

                #Save model
                model.save(f'{results_dir}/model/global_class_{num_classes}_ds{n_ds}_nch{n_ch}_T{T}_split_{split_ctr}.h5')

                #Clear Models
                K.clear_session()
                split_ctr = split_ctr + 1

            print('AVG \t {:.4f}\t{:.4f}'.format(acc[:,0].mean(), acc[:,1].mean()))


           


