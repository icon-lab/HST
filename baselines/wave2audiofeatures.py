# -*- coding: utf-8 -*-
"""
This is the pipeline for extracting features (MFCC, chroma, spectral center, spectral contrast) from audio waveforms for LSTM baselines.
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

warnings.filterwarnings("ignore")

SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
MFCC_dim = 13  # the MFCC dimension

label_dict = {
    "covidandroidnocough": 1,
    "covidandroidwithcough": 2,
    "covidwebnocough": 3,
    "covidwebwithcough": 4,
    "healthyandroidnosymp": -1,
    "healthyandroidwithcough": -2,
    "healthywebnosymp": -3,
    "healthywebwithcough": -4,
    "asthmaandroidwithcough": 6,
    "asthmawebwithcough": 8
}



def pad_audio(data, len_input=22050*30):
    if len(data) > len_input:
        offsetmax = len(data) - len_input
        offset = np.random.randint(offsetmax)
        data = data[offset : (len_input + offset)]

    else:
        if len_input > len(data):
            offsetmax = len_input - len(data)
            offset = np.random.randint(offsetmax)
        else:
            offset = 0
        data = np.pad(data, (offset, len_input - len(data) - offset), "constant")
    return data

def extract_mfcc(signal, signal_sr=SR, n_fft=FRAME_LEN, hop_length=HOP, n_mfcc=MFCC_dim):
    # compute the mfcc of the input signal
    mfcc = librosa.feature.mfcc(
        y=signal, sr=signal_sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc, dct_type=3)
    spectral_center = librosa.feature.spectral_centroid(
                y=signal, sr=signal_sr, hop_length=hop_length)
    chroma = librosa.feature.chroma_stft(y=signal, sr=signal_sr, hop_length=hop_length)
    spectral_contrast = librosa.feature.spectral_contrast(
        y=signal, sr=signal_sr, hop_length=hop_length)
    mfccs = np.zeros((mfcc.shape[1], 59))
    # extract the first and second order deltas from the retrieved mfcc's
    mfcc_delta = librosa.feature.delta(mfcc, order=1)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mfccs[:, 0:13] = mfcc.T
    mfccs[:, 13:26] = mfcc_delta.T
    mfccs[:, 26:39] = mfcc_delta2.T
    mfccs[:, 39:40] = spectral_center.T
    mfccs[:, 40:52] = chroma.T
    mfccs[:, 52:59] = spectral_contrast.T
    return mfccs.T
    
def get_resort(files):
    name_dict = {}
    for sample in files:
        type,name,others = sample.split('_',2)  # the UID is a mixed of upper and lower characters
        name = name.lower()
        name_dict['_'.join([type,name,others])] = sample
    re_file = [name_dict[s] for s in sorted(name_dict.keys())]
    return re_file

def extract_features(signal):
    signal = signal / np.max(np.abs(signal))
    trimmed_signal, idc = librosa.effects.trim(signal, frame_length=FRAME_LEN, hop_length=HOP)
    trimmed_signal = pad_audio(trimmed_signal)
    mfccs = extract_mfcc(trimmed_signal)
    return mfccs

if __name__ == "__main__":

    path = os.path.join(sys.argv[1])  # Data path
    #sys.argv[1]  # data path
    meta_breath2cough = json.load(open(os.path.join(path,"android_breath2cough.json")))
    
    #feature extraction 
    x_data = []
    y_label = []
    y_uid = []

    #first extract features for samples from [Android] 
    file_path = os.listdir(os.path.join(path,"data"))
    path = os.path.join(path,"data")
    for files in sorted(file_path):
        print(files)
        if files not in [
            "covidandroidnocough",
            "covidandroidwithcough",
            "healthyandroidwithcough",
            "healthyandroidnosymp",
            "asthmaandroidwithcough",]:
            continue
        sample_path = os.path.join(path,files,"breath")
        samples = os.listdir(sample_path)
        for sample in get_resort(samples):
            if ".wav_aug" in sample or "mono" in sample:
                continue
            print(sample)

            # for breath
            sample_path = os.path.join(path,files,"breath")
            file_b = os.path.join(sample_path,sample)

            name, filetpye = file_b.split(".")
            soundtype = sample.split("_")[0]
            if soundtype != "breaths":
                continue
            if filetpye != "wav":
                continue

            #remove the beginning and ending silence
            y, sr = librosa.load(file_b, sr=SR, mono=True, offset=0.0, duration=None)
            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
           
            features_breath = extract_features(signal=y)

            # for cough
            if sample in meta_breath2cough:
                sample_c = meta_breath2cough[sample]
                sample_path = os.path.join(path,files,"cough")
                file_c = os.path.join(sample_path,sample_c)
                try:
                    y, sr = librosa.load(
                        file_c, sr=SR, mono=True, offset=0.0, duration=None
                    )
                except IOError:
                    print("adroid cough doesn't exit")
                    continue
                else:
                    print("load")

                yt, index = librosa.effects.trim(
                    y, frame_length=FRAME_LEN, hop_length=HOP
                )
                duration = librosa.get_duration(y=yt, sr=sr)
                if duration < 2:
                    print("cough too short")
                    continue
                features_cough = extract_features(signal=y)
                                
                #combine breath and cough features for future use
                label = label_dict[files]
                uid = sample.split("_")[1]
                features = np.concatenate((features_breath, features_cough),
                                          axis=1
                )
                kk = features.tolist()
                x_data.append(features)
                y_label.append(label)
                y_uid.append(uid)
                print("save features!")
                
    #then extract features for samples from [Web]
    file_path = os.listdir(path)
    for fold in sorted(file_path): 
        print(fold)
        if fold not in [
            "covidwebnocough",
            "covidwebwithcough",
            "healthywebwithcough",
            "healthywebnosymp",
            "asthmawebwithcough",
            ]:
            continue
        fold_path = os.listdir(os.path.join(path,fold))
        for files in sorted(fold_path):
            print(files)
            # for breathe
            try:
                sample_path = (os.path.join(path,fold,files,"audio_file_breathe.wav"))
                file_b = sample_path
                y, sr = librosa.load(
                    file_b, sr=SR, mono=True, offset=0.0, duration=None
                )
            except IOError:
                print("breath doesn't exit")
                continue
            else:
                print("load")

            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("breath too short")
                continue  
            features_breath2 = extract_features(signal=y)

            # for cough
            try:
                sample_path = (os.path.join(path,fold,files,"audio_file_cough.wav"))
                file_c = sample_path
                y, sr = librosa.load(
                    file_c, sr=SR, mono=True, offset=0.0, duration=None
                )
            except IOError:
                print("cough doesn't exit")
                continue
            else:
                print("load")

            yt, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=yt, sr=sr)
            if duration < 2:
                print("cough too short")
                continue 
            features_cough2 = extract_features(signal=y)
            
            #combine breath and cough
            label = label_dict[fold]
            features2 = np.concatenate((features_breath2, features_cough2), 
                                      axis=1
            )
            x_data.append(features2)
            y_label.append(label)
            y_uid.append(files)
            print("save features")
    x_data = np.array(x_data)
    y_label = np.array(y_label)
    y_uid = np.array(y_uid)

    #save features in numpy.array
    np.save("x_data_audiofeat.npy", x_data)
    np.save("y_label_audio.npy", y_label)
    np.save("y_uid_audio.npy", y_uid)
