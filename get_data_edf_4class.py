#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Loads '.edf' data format in the data shape of EEGNet
"""

import numpy as np

# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib as edf
import os


__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

def get_data(subjects,runs,PATH):
    '''
    Keyword arguments:
    subject -- array of subject numbers in range [1, .. , 109] (integer)
    runs -- array of the numbers of the runs in range [1, .. , 14] (integer)
    
    Return: data_return     numpy matrix     size = NO_events x 64 x 656
            class_return    numpy matrix     size = NO_events
            
    '''
    """
    DATA EXPLANATION:
        
        LABELS: 
        both first_set and second_set
            T0: rest
        first_set (real motion in runs 3, 7, and 11; imagined motion in runs 4, 8, and 12)
            T1: the left fist 
            T2: the right fist
        second_set (real motion in runs 5, 9, and 13; imagined motion in runs 6, 10, and 14)
            T1: both fists
            T2: both feet
        
        This Program:
            Here, we get data from the first_set (rest, left fist, right fist), 
            and also data from the second_set (rest, both feet).
            We ignore data for T1 from the second_set and thus return data for 
            four classes/categories of events: Rest, Left Fist, Right Fist, Both Feet.
    """
    first_set = np.array([3,4,7,8,11,12])
    second_set = np.array([5,6,9,10,13,14])
    #first_set_real = np.array([3,7,11])
    #first_set_imagined = np.array([4,8,12])
    #second_set_real = np.array([5,9,13])
    #second_set_imagined = np.array([6,10,14])
    
    NO_channels = 64
    
    # depending on data
    NO_trials = 30 # each rung has 30 trials
    Window_Length = int(160 * 4.1)  # 656
    
    data_step = np.zeros((NO_trials,NO_channels,Window_Length))
    
    # initialize empty arrays to concatanate with itself later
    data_return = np.empty((0,NO_channels,Window_Length))
    class_return = np.empty(0)
    
    
    for subject in subjects:
        for run in runs:
            # Correct Naming for files
            if subject < 10:
                str_subject = '00'+str(subject)
            elif subject < 100:
                str_subject = '0'+str(subject)
            else:
                str_subject = str(subject)
                
            if run < 10:
                str_run = '0'+str(run)
            else:
                str_run = str(run)
            
            # Create file name variable to access edf file
            file_name = os.path.join(PATH+'../files/'+'S'+str_subject,'S'+str_subject+'R'+str_run+'.edf')
            # Read file
            f = edf.EdfReader(file_name)
            # Signal Parameters - measurement frequency
            freq = f.getSampleFrequency(0)
            # Number of eeg channels = number of signals in file
            n = f.signals_in_file
            # These are the eeg channel/electrode names
            #signal_labels = f.getSignalLabels()
            # Initiate 64*20000 matrix to hold all datapoints
            sigbufs = np.zeros((n, f.getNSamples()[0]))
            # Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
            for i in np.arange(n):
                # Save the read data (vectorwise) in a matrix
                # Fill the matrix with all datapoints from each channel
                sigbufs[i, :] = f.readSignal(i)
            
            # Get Label information
            annotations = f.readAnnotations()
            
            # close the file
            f.close()
            
            # Get the specific label information
            labels = annotations[2]
            points = freq*annotations[0]
            
            # Turn the labels into an integer array
            labels_int = np.zeros(np.size(labels))
            
            # initialize empty array for just feet data because it is variable in size for each run
            labels_for_feet = np.empty(0)
            data_feet = np.empty((0,NO_channels,Window_Length))
            
            
            if run in second_set:
                for ii in range (0,np.size(labels)):
                    if labels[ii] == 'T0':
                        labels_for_feet = np.append(labels_for_feet,[0])
                        # change data shape and seperate events
                        data_feet = np.vstack((data_feet, np.array(sigbufs[:,int(points[ii]):int(points[ii])+Window_Length])[None]))
                    if labels[ii] == 'T2':
                        labels_for_feet = np.append(labels_for_feet,[3])
                        # change data shape and seperate events
                        data_feet = np.vstack((data_feet, np.array(sigbufs[:,int(points[ii]):int(points[ii])+Window_Length])[None]))            
                   # concatenate arrays in order to get the whole data in one input array    
                data_return = np.concatenate((data_return,data_feet))
                class_return = np.concatenate((class_return,labels_for_feet)) 
            else:
                for ii in range (0,np.size(labels)):
                    if labels[ii] == 'T0':
                        labels_int[ii] = 0
                    if labels[ii] == 'T1':
                        labels_int[ii] = 1
                    if labels[ii] == 'T2':
                        labels_int[ii] = 2
                    # change data shape and seperate events
                    data_step[ii,:,:] = sigbufs[:,int(points[ii]):int(points[ii])+Window_Length]
                
                # concatenate arrays in order to get the whole data in one input array    
                data_return = np.concatenate((data_return,data_step))
                class_return = np.concatenate((class_return,labels_int))
            
            
    return data_return, class_return

