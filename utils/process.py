import librosa
import librosa.display
import numpy as np
import pandas as pd 
from sklearn.mixture import *
import matplotlib.pyplot as plt

def mfcc(wavData, frameRate, vad):
    mfcc = librosa.feature.mfcc(wavData, sr=16000, n_mfcc=20,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    else:
        print("Shapes are now equal")
    mfcc = mfcc[vad,:]
    return mfcc

def num_comp(mfcc):
    n_components = np.arange(1, 20)
    models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(mfcc) for n in n_components]
    plt.figure(figsize=(8, 6))
    plt.plot(n_components, [m.bic(mfcc) for m in models], label='BIC')
    plt.plot(n_components, [m.aic(mfcc) for m in models], label='AIC')
    plt.legend(loc='best')
    plt.xlabel('GMM n_components for an audio file')

    comp = pd.DataFrame()
    comp['BIC'] = [m.bic(mfcc) for m in models]
    comp['n_components'] = n_components
    num_comp = int(comp['n_components'][comp['BIC'] == comp['BIC'].min()])
    print("Minimum number of components with least value of BIC is", num_comp)

    return num_comp
