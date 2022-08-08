# -*- coding: utf-8 -*-
"""
This is the training pipeline for the LSTM baseline. 
x_data_audiofeat.npy, y_label_audio.npy, and y_uid_audio.npy must be previously obtained by running wave2audiofeatures.py.
"""
import json
import os
import sys
import warnings
from math import pi

import librosa
import numpy as np
import pandas as pd
from scipy.fftpack import fft, hilbert
from sklearn.preprocessing import scale

import random
import warnings

import numpy as np
from sklearn import decomposition
from sklearn import metrics
from sklearn import preprocessing

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupShuffleSplit

import logging
import os
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler

def seed_env(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class ScaleStandard(TransformerMixin):  #StandardScaler for 3D, taken from: https://stackoverflow.com/a/53231071
    def __init__(self, **kwargs):
        self._scaler = StandardScaler(copy=True)
        self._orig_shape = None

    def fit(self, X):
        X = np.array(X)
        if len(X.shape) > 1:
            self._orig_shape = X.shape[1:]
        X = self._flatten(X)
        self._scaler.fit(X)
        return self

    def transform(self, X):
        X = np.array(X)
        X = self._flatten(X)
        X = self._scaler.transform(X)
        X = self._reshape(X)
        return X

    def _flatten(self, X):
        if len(X.shape) > 2:
            n_dims = np.prod(self._orig_shape)
            X = X.reshape(-1, n_dims)
        return X

    def _reshape(self, X):
        if len(X.shape) >= 2:
            X = X.reshape(-1, *self._orig_shape)
        return X

warnings.filterwarnings("ignore")

def RandomUnderSampler(np_data, np_label): 
    """downsample the majority class according to the given labels.
    :param np_data: extracted features as a array
    :type np_data: numpy.ndarray
    :param np_label: correspoinds labes as a vector
    :type np_data: numpy.ndarray
    :return: feature vectors and labels for balanced samples
    :rtype: numpy.ndarray
    """
    label = list(set(np_label))
    
    # perform a sanity check
    if len(label) < 2:
        raise ValueError("Less than two classed input")
        
    # seperate two class
    number_c0 = np.sum(np_label == label[0])
    number_c1 = np.sum(np_label == label[1])
    x_c0 = np_data[np_label == label[0], :]
    x_c1 = np_data[np_label == label[1], :]
    y_c0 = np_label[np_label == label[0]]
    y_c1 = np_label[np_label == label[1]]
    
    # downsample the majority class
    random.seed(0)
    if number_c0 < number_c1:
        index = random.sample(range(0, number_c1), number_c0)
        x_c1 = x_c1[index, :]
        y_c1 = y_c1[index]

    else:
        index = random.sample(range(0, number_c0), number_c1)
        x_c0 = x_c0[index, :]
        y_c0 = y_c0[index]
        
    new_data = np.concatenate((x_c0, x_c1), axis=0)
    new_label = np.concatenate((y_c0, y_c1), axis=0)
    
    #return the balanced class
    return new_data, new_label

if __name__ == "__main__":
    x_data_all = np.load("x_data_audiofeat.npy", allow_pickle=True)
    y_label_all = np.load("y_label_audio.npy", allow_pickle=True)
    y_uid_all = np.load("y_uid_audio.npy", allow_pickle=True)


    # split the data for different tasks  
    x_data_all_1 = x_data_all[y_label_all == 1]  #covidandroidnocough
    x_data_all_2 = x_data_all[y_label_all == 2]  #covidandroidwithcough
    x_data_all_3 = x_data_all[y_label_all == 3]  #covidwebnocough
    x_data_all_4 = x_data_all[y_label_all == 4]  #covidwebwithcough
    x_data_all_6 = x_data_all[y_label_all == 6]  #asthmaandroidwithcough
    x_data_all_8 = x_data_all[y_label_all == 8]  #asthmawebwithcough
    x_data_all_m1 = x_data_all[y_label_all == -1] #healthyandroidnosymp
    x_data_all_m2 = x_data_all[y_label_all == -2] #healthyandroidwithcough
    x_data_all_m3 = x_data_all[y_label_all == -3] #healthywebnosymp
    x_data_all_m4 = x_data_all[y_label_all == -4] #healthywebwithcough

    y_label_all_1 = y_label_all[y_label_all == 1]
    y_label_all_2 = y_label_all[y_label_all == 2]
    y_label_all_3 = y_label_all[y_label_all == 3]
    y_label_all_4 = y_label_all[y_label_all == 4]
    y_label_all_6 = y_label_all[y_label_all == 6]
    y_label_all_8 = y_label_all[y_label_all == 8]
    y_label_all_m1 = y_label_all[y_label_all == -1]
    y_label_all_m2 = y_label_all[y_label_all == -2]
    y_label_all_m3 = y_label_all[y_label_all == -3]
    y_label_all_m4 = y_label_all[y_label_all == -4]

    y_uid_1 = y_uid_all[y_label_all == 1]
    y_uid_2 = y_uid_all[y_label_all == 2]
    y_uid_3 = y_uid_all[y_label_all == 3]
    y_uid_4 = y_uid_all[y_label_all == 4]
    y_uid_6 = y_uid_all[y_label_all == 6]
    y_uid_8 = y_uid_all[y_label_all == 8]
    y_uid_m1 = y_uid_all[y_label_all == -1]
    y_uid_m2 = y_uid_all[y_label_all == -2]
    y_uid_m3 = y_uid_all[y_label_all == -3]
    y_uid_m4 = y_uid_all[y_label_all == -4]

    # save csv for each group of experiments
    output = open("tasks_lstm.csv", "wb")
    head = [
        "Tasks",
        "Train",
        "Test",
        "Breathing_AUC",
        "Breathing_ACC",
        "Breathing_Pre",
        "Breathing_Rec",
        "Breathing_f1",
        "Cough_AUC",
        "Cough_ACC",
        "Cough_Pre",
        "Cough_Rec",
        "Cough_f1",
        "BreathingCough_AUC",
        "BreathingCough_ACC",
        "BreathingCough_Pre",
        "BreathingCough_Rec",
        "BreathingCough_F1",
    ]

    output.write(",".join(head).encode(encoding="utf-8"))
    output.write("\n".encode(encoding="utf-8"))
    j = 0
    for i1 in ["task1"]:
        print("Conduct", i1)
        line = ['balance' + i1]

        if i1 == "task1":
            # using sample from:
            # positive:covidandroidnocough + covidandroidwithcough + covidwebnocough + covidwebwithcough
            # negaive: healthyandroidnosymp + healthywebnosymp
            x_data_all_task = np.concatenate(
                (
                    x_data_all_1,
                    x_data_all_2,
                    x_data_all_3,
                    x_data_all_4,
                    x_data_all_m1,
                    x_data_all_m2,
                    x_data_all_m3,
                    x_data_all_m4
                ),
                axis=0,
            )
            y_label_all_task = np.concatenate(
                (
                    y_label_all_1,
                    y_label_all_2,
                    y_label_all_3,
                    y_label_all_4,
                    y_label_all_m1,
                    y_label_all_m2,
                    y_label_all_m3,
                    y_label_all_m4
                ),
                axis=0,
            )
            y_uid_all_task = np.concatenate(
                (y_uid_1, y_uid_2, y_uid_3, y_uid_4, y_uid_m1, y_uid_m2, y_uid_m3, y_uid_m4),
                axis=0
            )

            y_label_all_task[y_label_all_task > 0] = 1  # covid positive
            y_label_all_task[y_label_all_task < 0] = 0

        if i1 == "task2":
            #using sample from: 
            #positive:covidandroidwithcough  + covidwebwithcough  
            #negaive: healthyandroidwithcough  + healthywebwithcough
            x_data_all_task = np.concatenate(
                (x_data_all_2, x_data_all_4, x_data_all_m2, x_data_all_m4),
                axis=0,
            )
            y_label_all_task = np.concatenate(
                (y_label_all_2, y_label_all_4, y_label_all_m2, y_label_all_m4),
                axis=0,
            )
            y_uid_all_task = np.concatenate(
                (y_uid_2, y_uid_4, y_uid_m2, y_uid_m4), axis=0
            )

            y_label_all_task[y_label_all_task > 0] = 1  # covid positive
            y_label_all_task[y_label_all_task < 0] = 0
 

        # test different modalities
        for i2 in ["breath_cough"]:  # multi-modal
            if i2 == "breath":
                x_data_all_this = x_data_all_task[:, :, :601]
                print(x_data_all_this.shape)
            if i2 == "cough":
                x_data_all_this = x_data_all_task[:, :, 601:]
                print(x_data_all_this.shape)
            if i2 == "breath_cough":
                x_data_all_this = x_data_all_task

            
            acc = []
            pre = []
            rec = []
            auc = []
            f1list = []
            prauc = []
            train_ratio = []
            test_ratio = []


            for seed in [1, 2, 5, 12, 40, 52, 72, 2002, 4002, 6002]:
                
                seed_env(seed)
                
                gss = GroupShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=seed
                )
                idx1, idx2 = next(
                    gss.split(x_data_all_this, groups=y_uid_all_task)
                )

                # Get the split DataFrames.
                train_x, test_x = x_data_all_this[idx1], x_data_all_this[idx2]
                y_train, y_test = y_label_all_task[idx1], y_label_all_task[idx2]
                uid_train, uid_test = y_uid_all_task[idx1], y_uid_all_task[idx2]

                train_x, y_train = RandomUnderSampler(train_x, y_train) 
                test_x, y_test = RandomUnderSampler(test_x, y_test)
                train_ratio.append(train_x.shape[0])
                test_ratio.append(test_x.shape[0])
                
                # scale data
                scaler = ScaleStandard()
                tensor = scaler.fit_transform(train_x)
                tensor2 = scaler.fit_transform(test_x)
                
                input_shape = (train_x.shape[1], train_x.shape[2])

                print("Build LSTM RNN model ...")
                
                model = Sequential()
                
                model.add(LSTM(units=32, dropout=0.3, activation= "tanh", input_shape=input_shape))       
                model.add(Dense(8, activation='relu'))
                model.add(Dropout(0.3))
                model.add(Dense(units=1, activation="sigmoid"))
                loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2, clipnorm=1.0) #try clip norm
                model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()])
                history = model.fit(tensor, y_train, epochs=25, batch_size=16, validation_data=(tensor2,y_test))
                score, accuracy, precision, recall, aucc = model.evaluate(tensor2, y_test, batch_size=1000, verbose=1)

                if j == 0:
                    max_pre = history.history['val_precision']
                    max_acc = history.history['val_accuracy']
                    max_auc = history.history['val_auc']
                    max_rec = history.history['val_recall']
                else:
                    max_pre = history.history['val_precision'+"_"+str(j)]
                    max_acc = history.history['val_accuracy']
                    max_auc = history.history['val_auc'+"_"+str(j)]
                    max_rec = history.history['val_recall'+"_"+str(j)]
                j += 1

                ind = max_auc.index(max(max_auc))
                print("Dev loss:  ", score)

                print("seed: "+str(seed)," auc:", max_auc[ind], " precision:", max_pre[ind], " recall", max_rec[ind])
                pre.append(max_pre[ind])
                acc.append(max_acc[ind])
                auc.append(max_auc[ind])
                rec.append(max_rec[ind])
                pre_rec = max_pre[ind]+max_rec[ind]
                if pre_rec == 0:
                    f1 = 0
                else:
                    f1 = 2*(max_pre[ind])*(max_rec[ind])/pre_rec
                f1list.append(f1)

            line.append(
                "{:.4f}".format(np.mean(auc)) + "("
                "{:.4f}".format(np.std(auc)) + ")"
            )
            line.append(
                "{:.4f}".format(np.mean(acc)) + "("
                "{:.4f}".format(np.std(acc)) + ")"
            )
            line.append(
                "{:.4f}".format(np.mean(pre)) + "("
                "{:.4f}".format(np.std(pre)) + ")"
            )
            line.append(
                "{:.4f}".format(np.mean(rec)) + "("
                "{:.4f}".format(np.std(rec)) + ")"
            )
            line.append(
                "{:.4f}".format(np.mean(f1list)) + "("
                "{:.4f}".format(np.std(f1list)) + ")"
            )
            
        output.write(",".join(line).encode(encoding="utf-8"))
        output.write("\n".encode(encoding="utf-8"))
        print("---------------")
            
