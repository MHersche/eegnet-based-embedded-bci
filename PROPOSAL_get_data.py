#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads '.edf' data format in the data shape of EEGNet
"""
__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import os
import numpy as np

# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib as edf

# to calculate mean and the standard deviation
import statistics as stats
import random

def get_data(PATH, long=False, subjects_list=range(1,110), n_classes=4, , preshuffle=False, expand_dims=False, labels_onehot=False, normalization=0):
    '''
    Keyword arguments:
    PATH: path to directory under which the test data lies.
    normalization -- [0 (default)]: no normalization, 1: normalized per channel, 2: normalized per all trials
    long -- If True: Trials of length 6s returned. If False: Trials of length 3s returned
    subjects_list -- [range(1,110) (default)]: array of subject numbers in range [1, .. , 109] 
            ,used if not all subjects are to be taken
    n_classes : number of classes of data
    
    Return: data_return     numpy matrix     size = NO_events x 64 x 656
            class_return    numpy matrix     size = NO_events
            X_Train: Training Trials
            y_Train: Training labels
            mean: (if normalization != 0) mean of normalized Training set
            std_dev: (if normalization != 0) std_dev of normalized Training set
    '''
    # Define subjects whose data is not taken, for details see data tester added 106 again to analyze it, deleted from the exluded list
    excluded_subjects = [88,92,100,104]
    # Define subjects whose data is taken, namely from 1 to 109 excluding excluded_subjects
    subjects = [x for x in subjects_list if (x not in excluded_subjects)]

    # changed the runs include baseline to get on subjects data
    mi_runs = [1, 4, 6, 8, 10, 12, 14]
    # Extract only requested number of classes
    if(n_classes == 3):
        print('Returning 3 Class data')
        mi_runs.remove(6)
        mi_runs.remove(10)
        mi_runs.remove(14)
    elif(n_classes == 2):
        print('Returning 2 Class data')
        mi_runs.remove(6)
        mi_runs.remove(10)
        mi_runs.remove(14)
        mi_runs.remove(1)
    print(f'Returning data from runs: {mi_runs}')
    
    # READ DATA
    X_Train, y_Train = read_data(subjects=subjects, runs=mi_runs, PATH=PATH, long=long)

    if(preshuffle):
        # Shuffle Data reproducably
        np.random.seed(42)
        np.random.shuffle(X_Train)
        np.random.seed(42)
        np.random.shuffle(y_Train)
        
    if(expand_dims):
        # Expand dimensions to match EEGNet input dimensions
        X_Train = (np.expand_dims(X_Train, axis=1))
        
    if(labels_onehot):
        # Convert integer labels to one-hot encodings
        y_Train  = np_utils.to_categorical(y_Train)
    
    # RETURN DATA
    if(normalization == 0):
        return X_Train, y_Train
    
    # do normalization
    if(normalization == 1):
        #TODO: declare std_dev, mean arrays to return
        for i in range(X_Train.shape[0]):
            for ii in range(X_Train.shape[1]):
                std_dev = stats.stdev(X_Train[i,ii])
                mean = stats.mean(X_Train[i,ii])
                X_Train[i,ii] = (X_Train[i,ii] - mean) / std_dev
        
        return X_Train, y_Train, mean, std_dev
    
    if(normalization == 2):
        #TODO: implement second type of normalization
        mean = std_dev = 0
        
        return X_Train, y_Train, mean, std_dev
    
    
def read_data(subjects, runs, PATH, long=False):
    '''
    Get Data from given subjects, runs in specified format.
    
    Keyword arguments:
    subject : array of subject numbers in range [1, .. , 109] (integer)
    runs    : array of the numbers of the runs in range [1, .. , 14] (integer)
    PATH    : base directory of .edf data from PhysioNet database
    long    : boolean of whether 3s or 6s data should be used. if long: WINDOW_LENGTH = 6s else: WINDOW_LENGTH = 3s
    
    Return: 
    X       :    numpy matrix     size = NO_events x NO_CHANNELS(64) x WINDOW_LENGTH(3s or 6s) * SAMPLE_FREQUENCY(160Hz)
    y       :     numpy matrix     size = NO_events
            
    
    DATA EXPLANATION:
    [Integer Labels]: Chosen to be consistent throughout changing number of classes of data to be returned
    all runs:
        T0: rest            [2]
    first_set (imagined motion in runs 4, 8, and 12)
        T1: the left fist   [0]
        T2: the right fist  [1]
    second_set (imagined motion in runs 6, 10, and 14)
        T1: both fists      [X] (not taken)
        T2: both feet       [3]
    '''
    # BASE STRINGS TO MODIFY FOR EACH SUBJECT TO ACCESS .edf FILES
    base_file_name = 'S{:03d}R{:02d}.edf'
    base_subject_directory = 'S{:03d}'
    
    # SETS OF RUNS WITH DIFFERENT CLASSES OF DATA
    baseline   = np.array([1])
    first_set  = np.array([4,8,12])
    second_set = np.array([6,10,14])
    
    # SIGNAL PARAMETERS
    
    # EEG Measurement frequency
    SAMPLE_FREQUENCY = 160
    # Number of EEG channels
    NO_CHANNELS = 64
    
    # Number of Trials extracted per Run
    NO_TRIALS = 7
    
    # Define Sample size per Trial 
    if not long:
        WINDOW_LENGTH = int(SAMPLE_FREQUENCY * 3) # 3s Trials: 480 samples
    else:
        WINDOW_LENGTH = int(SAMPLE_FREQUENCY * 6) # 6s Trials: 960 samples 
    
    # initialize empty arrays to concatenate with itself later
    X = np.empty((0,NO_CHANNELS,WINDOW_LENGTH))
    y = np.empty(0)
    
    for subject in subjects:

        for run in runs:
            # CLASS COUNTERS
            #For each run, a certain number of trials from corresponding classes should be extracted
            counter_0 = 0
            counter_L = 0
            counter_R = 0
            counter_F = 0
            
            # READ DATA
            # Create file name variable to access .edf file
            filename = base_file_name.format(subject,run)
            directory = base_subject_directory.format(subject)
            file_name = os.path.join(PATH,directory,filename)
            
            # Get Signal
            f = edf.EdfReader(file_name)
            sigbufs = np.zeros((NO_CHANNELS, f.getNSamples()[0]))
            # Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
            for i in np.arange(NO_CHANNELS):
                # Save the read data (vectorwise) in a matrix
                # Fill the matrix with all datapoints from each channel
                sigbufs[i, :] = f.readSignal(i)
            
            # READ LABELS
            # Get Label information
            annotations = f.readAnnotations()
            # close the file
            f.close()
            
            # Get the specific label information
            labels = annotations[2]
            points = SAMPLE_FREQUENCY*annotations[0]
            
            # TEMPORARY ARRAYS FOR STORING DATA
            labels_int = np.empty(0)
            data_step = np.empty((0,NO_CHANNELS, WINDOW_LENGTH))             
            
            # SAVE DATA, LABELS
            # First set runs include MI data from classes: Left Fist (L), Right Fist (R), Rest (0).
            # Only L and R data is extracted from this set of runs
            if run in first_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_TRIALS):
                        continue
                        #counter_0 += 1
                        #labels_int = np.append(labels_int, [2])
                        
                    elif(labels[ii] == 'T1' and counter_L < NO_TRIALS):
                        counter_L += 1
                        labels_int = np.append(labels_int, [0])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+WINDOW_LENGTH])[None]))
                        
                    elif(labels[ii] == 'T2' and counter_R < NO_TRIALS):
                        counter_R += 1
                        labels_int = np.append(labels_int, [1])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+WINDOW_LENGTH])[None]))
            
            # Second set runs include MI data from classes: Both Feet (F), Rest (0).
            # Only F data is extracted from this set of runs
            elif run in second_set:
                for ii in range(0,np.size(labels)):
                    if(labels[ii] == 'T0' and counter_0 < NO_TRIALS):
                        continue
                        #counter_0 += 1
                        #labels_int = np.append(labels_int,[2])
                        
                    elif(labels[ii] == 'T2' and counter_F < NO_TRIALS):
                        counter_F += 1
                        labels_int = np.append(labels_int,[3])
                        # change data shape and seperate events
                        data_step = np.vstack((data_step, np.array(sigbufs[:,int(points[ii]):int(points[ii])+WINDOW_LENGTH])[None]))        
            
            # The Baseline run includes MI data from class: Rest (0).
            # 0 data is extracted from this run
            elif run in baseline:
                # Extract 20 Non-overlapping Trials from first 60 seconds
                for ii in range(0,20):
                    if(counter_0 < 20):  
                        counter_0 += 1
                        labels_int = np.append(labels_int, [2])
                        data_step = np.vstack((data_step, np.array(sigbufs[:,(ii*WINDOW_LENGTH):((ii+1)*WINDOW_LENGTH)])[None]))
                random.seed(42)
                # Add one more Trial to achieve 21 in total
                index = random.randint(0,57*SAMPLE_FREQUENCY)
                labels_int = np.append(labels_int, [2])
                data_step = np.vstack((data_step, np.array(sigbufs[:,(index):(index+WINDOW_LENGTH)])[None]))
                '''
                if train == True:
                    data_step = data_step[0:14]
                    labels_int = labels_int[0:14]
                else:
                    data_step = data_step[14:21]
                    labels_int = labels_int[14:21]
                '''
            # concatenate arrays in order to get the whole data in one input array    
            X = np.concatenate((X,data_step))
            y = np.concatenate((y,labels_int))
        
    return X, y

