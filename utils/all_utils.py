from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

def show_plot(mix_sound):
    """Plots the audio file 

    Args:
        mix_sound (.wav): Audio Files
    """
    
    samplerate, data = read(mix_sound)
    print("Sample rate:",samplerate)
    duration = len(data)/samplerate
    print("Duration of Audio in Seconds", duration)

    time = np.arange(0,duration,1/samplerate)

    plt.figure(figsize = (8,6))
    plt.plot(time,data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('6TU5302374.wav')
    plt.show()
    plt.savefig('plot_audio.png')


def VoiceActivityDetection(wavData, frameRate):
    """Used to trim silences in the audio file 

    Args:
        wavData (.wav): audio file in .wav format
        frameRate (int): predefined frame rate 

    Returns:
        wavdata: returns wave data file by trimming silences
    """
    # uses the librosa library to compute short-term energy
    ste = librosa.feature.rms(wavData,hop_length=int(16000/frameRate)).T
    thresh = 0.1*(np.percentile(ste,97.5) + 9*np.percentile(ste,2.5))    # Trim 5% off and set threshold as 0.1x of the ste range
    return (ste>thresh).astype('bool')



def kmeans(X_normalized):
    # Elbow Method for K means
    # Import ElbowVisualizer
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2,6), metric='calinski_harabasz', timings= True)
    visualizer.fit(X_normalized)        # Fit data to visualizer
    visualizer.show()
    num_clus_k = visualizer.elbow_value_
    return num_clus_k



