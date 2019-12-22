#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import os
import sys

sys.path.insert(1, '../')

# our functions to test and load data
import get_data as get

PATH = "../../files/"
# Load data
X_all_Train, y_Train = get.get_data(PATH,n_classes = 4)

shape = np.shape(X_all_Train)

reshaped = np.reshape(X_all_Train, (shape[0],shape[1] * shape[2]))

# np.savetxt('../../signals_csv/data_4class.csv', reshaped)
np.savetxt('../../signals_csv/data_4class_labels.csv', y_Train)

