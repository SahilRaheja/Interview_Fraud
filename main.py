import pandas as pd
import numpy as np 
import librosa
from utils.all_utils import show_plot, VoiceActivityDetection, kmeans
from utils.process import mfcc, num_comp
from utils.model import trainGMM, clust
from sklearn.preprocessing import StandardScaler, normalize
import logging

logging_str = "[%(asctime)s: ]"


def main(mix_sound):
    show_plot(mix_sound)
    wavData,sr = librosa.load(mix_sound, sr = 16000)
    vad=VoiceActivityDetection(wavData,frameRate)

    mfcc = mfcc(wavData, frameRate, vad)
    num_comp = num_comp(mfcc)

    clusterset = trainGMM(mix_sound, frameRate, segLen, vad, numMix, num_comp)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clusterset)  
    # Normalizing the data so that the data approximately follows a Gaussian distribution
    X_normalized = normalize(X_scaled)

    num_cluster = kmeans(X_normalized)

    print('Number of Speakers are {}'.format(num_cluster))

    cluster = clust(num_cluster, X_normalized)

if __name__=='__main__':
    segLen,frameRate,numMix = 3,50,128
    mix_sound = r'C:\Users\Pranav\Projects\Voice\Interview\Interview Recordings Sample Dataset\Mix_Sahil_Rajeev_Rahul.wav'
    main(mix_sound)

