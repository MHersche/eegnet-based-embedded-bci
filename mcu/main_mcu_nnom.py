#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
import sys

sys.path.insert(1, '../nnom/scripts/')
sys.path.insert(1, '../new_models/')

from datetime import datetime
# pyEDFlib is a python library to read/write EDF+/BDF+ files based on EDFlib.
import pyedflib
# our functions to test and load data
import get_data as get
#import data_tester as test
# tensorflow part
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import optimizers
from keras.optimizers import Adam
from keras.models import load_model
# EEGNet models
import models as models
from nnom_utils import *

model = load_model('models/4class_19ch_freqfactor5.h5')

X_all_Train, y_Train = get.get_data(PATH,n_classes = 4)

x_test = X_all_Train[7056:,:,:]
y_Train = y_Train[7056:]
y_test      = np_utils.to_categorical(y_Train)
X_Train_real = (np.expand_dims(X_Train, axis=1))

generate_test_bin(X_Train_real*127, y_Train_cat, name='test_data.bin')

scores = evaluate_model(model,x_test, y_test)

generate_model(model, x_test, format='hwc', name="weights.h" )



