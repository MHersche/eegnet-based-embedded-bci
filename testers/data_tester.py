# -*- coding: utf-8 -*-
__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

"""
    Program to test given data. Looks for variations in the signal properties and reports them back to the user.
    
    USAGE: tune file_name line to look for correct data locations, change default values for signal parameters
    example usage:
        PATH = "../files/"
        s = [i for i in range(1,110)]
        r = [i for i in range(3,15)]
        subs = test_data(s, r, PATH, 160, 64, 30) #The last three arguments are optional, these are the default values.

"""

# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
import numpy as np
import os

def test_data(subjects, runs, PATH, default_frequency=160, default_no_channels=64, default_no_events=30):
    
    '''
    Keyword arguments:
    subjects -- array of subject numbers in range [1, .. , 109] (integer)
    runs     -- array of the numbers of the runs in range [1, .. , 14] (integer)
    PATH     -- string of PATH to directory containing .edf data
                ex: /home/<username>/<project_directory>/<data>/
                to this, SXXXRYY.edf will be added to form the full path

    Return:
    subjects -- array of subjects who were not removed
    '''
    # Define expected values
    frequency = default_frequency       #160 [Hz]
    eeg_channels = default_no_channels  #60
    events = default_no_events          #30
    
    for subject in subjects:
        for run in runs:
            
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
            file_name = os.path.join(PATH +'S' + subject_str,'S'+ subject_str + 'R' + run_str + '.edf')
            
            # Read file
            f = pyedflib.EdfReader(file_name)
                
            # Signal Parameters
            # Number of eeg channels = number of signals in file
            n = f.signals_in_file
            # Get Label information
            annotations = f.readAnnotations()
            labels = annotations[2]
            # Measurement Frequency
            freq = f.getSampleFrequencies()
                        
            # Check irregularities
            for i in np.arange(1, len(freq)):
                if (freq[i-1] != freq[i]):
                    print('S'+ subject_str + 'R' + run_str + " contains channels with differing frequencies!")
                    if subject in subjects: subjects.remove(subject)            
            
            if (frequency != freq[0]):
                print('S'+ subject_str + 'R' + run_str + "  removed: Frequency different: " + str(frequency) + " to: " + str(freq[0]))
                if subject in subjects: subjects.remove(subject)
            
            if (eeg_channels != n):
                print('S'+ subject_str + 'R' + run_str + "  Number of signals different: " + str(eeg_channels) + " to: " + str(n))
                if subject in subjects: subjects.remove(subject)
                
            if (events != len(labels)):
                print('S'+ subject_str + 'R' + run_str + "  Number of Events per Run different: " + str(events) + " to: " + str(len(labels)))
                if subject in subjects: subjects.remove(subject)
                
            f.close()
    return subjects