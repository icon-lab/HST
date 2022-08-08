"""
This is the preprocessing pipeline for converting the waveforms into spectrogram images.

For Task 1,
#positive:covidandroidnocough + covidandroidwithcough + covidwebnocough + covidwebwithcough  
#negaive: healthyandroidnosymp + healthywebnosymp

For Task 2,
#positive:covidandroidwithcough  + covidwebwithcough  
#negaive: healthyandroidwithcough  + healthywebwithcough

"""

import torch
import torchvision
import random
import numpy as np
import librosa
import librosa.display
import pandas as pd
import os
from PIL import Image
import pathlib
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from tqdm.autonotebook import tqdm
import torchvision.transforms as T


#For android covid data
data_paths = ['data/covidandroidwithcough']
cmap = plt.get_cmap('gray')
plt.figure(figsize=(8,8))
SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
audio_type = 'breath cough'.split()
for data_path in data_paths:
    for g in audio_type:
        pathlib.Path(f'covidandroidcough/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'{data_path}/{g}'):
            sample_name = f'{data_path}/{g}/{filename}'
            y, sr = librosa.load(sample_name,sr=SR, mono=True, offset=0.0, duration=None)
            y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=y, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
            plt.specgram(y, NFFT=2048, Fs=SR, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'covidandroidcough/{g}/{filename[:-3].replace(".", "")}.png')
            plt.clf()

#For android healthy data
data_paths = ['data/healthyandroidwithcough']
cmap = plt.get_cmap('gray')
plt.figure(figsize=(8,8))
SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
audio_type = 'breath cough'.split()
for data_path in data_paths:
    for g in audio_type:
        pathlib.Path(f'healthyandroidcough/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'{data_path}/{g}'):
            sample_name = f'{data_path}/{g}/{filename}'
            y, sr = librosa.load(sample_name,sr=SR, mono=True, offset=0.0, duration=None)
            y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=y, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'/healthyandroidcough/{g}/{filename}.png')
            plt.clf()            

#For android covid data without cough symptoms
data_paths = ['data/covidandroidnocough']
cmap = plt.get_cmap('gray')
plt.figure(figsize=(8,8))
SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
audio_type = 'breath cough'.split()
for data_path in data_paths:
    for g in audio_type:
        pathlib.Path(f'covidandroidwithoutcough/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'{data_path}/{g}'):
            sample_name = f'{data_path}/{g}/{filename}'
            y, sr = librosa.load(sample_name,sr=SR, mono=True, offset=0.0, duration=None)
            y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=y, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'/covidandroidwithoutcough/{g}/{filename}.png')
            plt.clf() 
            
            
#For android healthy data without cough symptoms
data_paths = ['data/healthyandroidnosymp']
cmap = plt.get_cmap('gray')
plt.figure(figsize=(8,8))
SR = 22050  # sample rate
FRAME_LEN = int(SR / 10)  # 100 ms
HOP = int(FRAME_LEN / 2)  # 50% overlap, meaning 5ms hop length
audio_type = 'breath cough'.split()
for data_path in data_paths:
    for g in audio_type:
        pathlib.Path(f'healthyandroidwithoutsymp/{g}').mkdir(parents=True, exist_ok=True)
        for filename in os.listdir(f'{data_path}/{g}'):
            sample_name = f'{data_path}/{g}/{filename}'
            y, sr = librosa.load(sample_name,sr=SR, mono=True, offset=0.0, duration=None)
            y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
            duration = librosa.get_duration(y=y, sr=sr)
            #filter out the audio shorter then 2 seconds. 
            if duration < 2:
                print("breath too short")
                continue
            plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
            plt.axis('off');
            plt.savefig(f'/healthyandroidwithoutsymp/{g}/{filename}.png')
            plt.clf() 
            
#then extract features for samples from [Web] covid
path = 'data/'
file_path = os.listdir(path)
for fold in sorted(file_path): 
    print(fold)
    if fold not in [
        "covidwebwithcough",
        ]:
        continue
    fold_path = os.listdir(os.path.join(path,fold))
    for files in sorted(fold_path):
        g = "breath"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("breath too short")
            continue 
        #save spectogram
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/covidwebcough/{g}/{filename}.png')
        plt.clf() 

        # for cough
        g = "cough"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("cough too short")
            continue
        #save spectogram
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/covidwebcough/{g}/{filename}.png')
        plt.clf() 
        print("saved spectogram")
 

#then extract features for samples from [Web] covid without cough symptoms
path = 'data/'
file_path = os.listdir(path)
for fold in sorted(file_path): 
    print(fold)
    if fold not in [
        "covidwebnocough",
        ]:
        continue
    fold_path = os.listdir(os.path.join(path,fold))
    for files in sorted(fold_path):
        g = "breath"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("breath too short")
            continue 
        #save spectogram
        healthywebwithcough_breath_files += 1
        healthywebwithcough_breath_duration += duration
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/content/covidwebwithoutcough/{g}/{files}.png')
        plt.clf()

        # for cough
        g = "cough"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("cough too short")
            continue
        #save spectogram
        healthywebwithcough_cough_files += 1
        healthywebwithcough_cough_duration += duration
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/content/covidwebwithoutcough/{g}/{files}.png')
        plt.clf()   
        print("saved spectogram")

#then extract features for samples from [Web] healthy
path = 'data/'
file_path = os.listdir(path)
for fold in sorted(file_path): 
    print(fold)
    if fold not in [
        "healthywebwithcough",
        ]:
        continue
    fold_path = os.listdir(os.path.join(path,fold))
    for files in sorted(fold_path):
        g = "breath"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("breath too short")
            continue 
        #save spectogram
        healthywebwithcough_breath_files += 1
        healthywebwithcough_breath_duration += duration
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/content/healthywebcough/{g}/{files}.png')
        plt.clf()

        # for cough
        g = "cough"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("cough too short")
            continue
        #save spectogram
        healthywebwithcough_cough_files += 1
        healthywebwithcough_cough_duration += duration
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/content/healthywebcough/{g}/{files}.png')
        plt.clf()   
        print("saved spectogram")
            
#then extract features for samples from [Web] healthy without any symptoms
path = 'data/'
file_path = os.listdir(path)
for fold in sorted(file_path): 
    print(fold)
    if fold not in [
        "healthywebnosymp",
        ]:
        continue
    fold_path = os.listdir(os.path.join(path,fold))
    for files in sorted(fold_path):
        g = "breath"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("breath too short")
            continue 
        #save spectogram
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/healthywebwithoutsymp/{g}/{filename}.png')
        plt.clf() 

        # for cough
        g = "cough"
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

        y, index = librosa.effects.trim(y, frame_length=FRAME_LEN, hop_length=HOP)
        duration = librosa.get_duration(y=y, sr=sr)
        if duration < 2:
            print("cough too short")
            continue
        #save spectogram
        plt.specgram(y, NFFT=2048, Fs=2, Fc=0, noverlap=128, cmap=cmap, sides='default', mode='default', scale='dB');
        plt.axis('off');
        plt.savefig(f'/healthywebwithoutsymp/{g}/{filename}.png')
        plt.clf() 
        print("saved spectogram")           
            
