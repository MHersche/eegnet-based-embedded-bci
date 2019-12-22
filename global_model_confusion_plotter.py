#!/usr/bin/env python3

__author__ = "Batuhan Tomekce and Burak Alp Kaya"
__email__ = "tbatuhan@ethz.ch, bukaya@ethz.ch"

import numpy as np
import pandas as pd
import get_data as get
from tensorflow.keras import utils as np_utils
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sn

# Remove excluded subjects from subjects list
def exclude_subjects(all_subjects=range(1,110), excluded_subjects=[88,92,100,104]):
    subjects = [x for x in all_subjects if (x not in excluded_subjects)]
    return subjects

# Set data path
PATH = "../files/"
# Make necessary directories for files
results_dir=f'conf_test'

# Specify number of classses for input data
num_classes_list = [2,3,4]
subjects = exclude_subjects()

for num_classes in num_classes_list:
    # using 5 folds
    num_splits = 5
    kf_global = KFold(n_splits = 5)
    n_epochs = 10

    split_ctr = 0
    for train_global, test_global in kf_global.split(subjects):
        if(split_ctr > 0):
            continue
        split_ctr += 1
        subs = [subjects[x] for x in test_global]
        X_sub, y_sub = get.get_data(PATH, n_classes=num_classes, subjects_list=subs)
        X_sub = np.expand_dims(X_sub, axis=1)
        y_sub_cat = np_utils.to_categorical(y_sub)
        SAMPLE_SIZE = np.shape(X_sub)[3]
        model = load_model(f'global_models/model/global_class_{num_classes}_split_0_v1.h5')

        probs = model.predict(X_sub)
        preds = probs.argmax(axis = -1)
        acc = np.mean(preds == y_sub)
        print(acc)
        true = y_sub.astype(int)
        pred = preds
        cm = confusion_matrix(y_sub,preds)
        if(num_classes == 2):
            classes= ['L', 'R']
        elif(num_classes == 3):
            classes = ['L', 'R','0']
        else:
            classes= ['L', 'R','0','F']
        df_cm = pd.DataFrame(cm, index = classes, columns = classes)
        ax = sn.heatmap(df_cm,annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(f'global_models/plots/conf_{num_classes}_global_0.pdf')

        df_cm_norm = df_cm /df_cm.sum(axis=1)
        ax = sn.heatmap(df_cm_norm,annot=True)
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)
        plt.savefig(f'global_models/plots/conf_norm_{num_classes}_global_0.pdf')
