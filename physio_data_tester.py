# -*- coding: utf-8 -*-
__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
import numpy as np
import os

frequency = 0
num_sig = 0
num_label = 0


for subject in np.arange(1,110):
    for run in np.arange(3,15):
        
        if(subject < 10):
            subject_str = '00'+str(subject)
        elif(subject < 100):
            subject_str = '0'+str(subject)
        else:
            subject_str = str(subject)
            
        if(run < 10):
            run_str = '0'+str(run)
        else:
            run_str = str(run)

        # Create file name variable to access edf file
        file_name = os.path.join('/home/bukaya/mlEEG/pyshio/files/S' + subject_str,'S'+ subject_str + 'R' + run_str + '.edf')
            
        # Read file
        f = pyedflib.EdfReader(file_name)
            
        # Signal Parameters - measurement frequency
        freq = f.getSampleFrequencies()
        for i in np.arange(1, len(freq)):
            if (freq[i-1] != freq[i]):
                print("This Run contains channels with differing frequencies!")
        
        # Number of eeg channels = number of signals in file
        n = f.signals_in_file
        
        # These are the eeg channel/electrode names
        signal_labels = f.getSignalLabels()

        # Get Label information
        annotations = f.readAnnotations()
        labels = annotations[2]        
        # Check irregularities
        if (frequency != freq[0]):
            print('S'+ subject_str + 'R' + run_str +"Frequency changed: " + str(frequency) + " to: " + str(freq[0]))
            frequency = freq[0]
        
        if (num_sig != n):
            print('S'+ subject_str + 'R' + run_str +"Number of signals changed: " + str(num_sig) + " to: " + str(n))
            num_sig = n
            
        if (num_label != len(labels)):
            print('S'+ subject_str + 'R' + run_str +"Number of Events per Run changed: " + str(num_label) + " to: " + str(len(labels)))
            num_label = len(labels)
            
        f.close()
f.close