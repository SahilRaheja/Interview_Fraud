from sklearn.mixture import *
import numpy as np 
import pandas as pd
import librosa
import librosa.display
from sklearn.cluster import AgglomerativeClustering


def trainGMM(wavFile, frameRate, segLen, vad, numMix, num_comp):
    wavData,_ = librosa.load(wavFile, sr = 16000)
    mfcc = librosa.feature.mfcc(wavData, sr=16000, n_mfcc=num_comp,hop_length=int(16000/frameRate)).T
    vad = np.reshape(vad,(len(vad),))
    if mfcc.shape[0] > vad.shape[0]:
        vad = np.hstack((vad,np.zeros(mfcc.shape[0] - vad.shape[0]).astype('bool'))).astype('bool')
    elif mfcc.shape[0] < vad.shape[0]:
        vad = vad[:mfcc.shape[0]]
    mfcc = mfcc[vad,:]
    print("Training GMM..")
    GMM = GaussianMixture(n_components=numMix, covariance_type='diag').fit(mfcc)
    segLikes = []
    segSize = frameRate*segLen
    for segI in range(int(np.ceil(float(mfcc.shape[0])/(frameRate*segLen)))):
        startI = segI*segSize
        endI = (segI+1)*segSize
#         print(startI)
#         print(endI)
        if endI > mfcc.shape[0]:
            endI = mfcc.shape[0]-1
        if endI==startI:    # Reached the end of file
            break
        seg = mfcc[startI:endI,:]
#         print(seg.shape[0])
        compLikes = np.sum(GMM.predict_proba(seg),0)
        segLikes.append(compLikes/seg.shape[0])
    print("Training Done")
    return np.asarray(segLikes)

def clust(num_clus_k, X_normalized):
    cluster = AgglomerativeClustering(n_clusters=num_clus_k, affinity='euclidean', linkage='ward') 
    clust=cluster.fit_predict(X_normalized)
    return clust