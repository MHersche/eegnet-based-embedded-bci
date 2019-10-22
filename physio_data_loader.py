# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
import numpy as np
import os


# Create file name variable to access edf file
file_name = os.path.join('/home/bukaya/mlEEG/pyshio/files/S001',
                         'S001R03.edf')
# Read file
f = pyedflib.EdfReader(file_name)
# Signal Parameters - measurement frequency
freq = f.getSampleFrequency(0)
# Number of eeg channels = number of signals in file
n = f.signals_in_file
# These are the eeg channel/electrode names
signal_labels = f.getSignalLabels()
# Initiate 64*20000 matrix to hold all datapoints
sigbufs = np.zeros((n, f.getNSamples()[0]))
# Here: n=64 arange(n) creates array ([0, 1, ..., n-2, n-1])
for i in np.arange(n):
    # Save the read data (vectorwise) in a matrix
    # Fill the matrix with all datapoints from each channel
    sigbufs[i, :] = f.readSignal(i)

# Get Label information
annotations = f.readAnnotations()
labels = annotations[2]
points = freq*annotations[0] 