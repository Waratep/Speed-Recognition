from keras.models import load_model
import librosa
import numpy as np
import pandas as pd
import multiprocessing
from joblib import Parallel, delayed
from librosa.feature import melspectrogram ,delta
from librosa.core import amplitude_to_db


model = load_model('switch.h5')

audio_path = 'dataset/test3.wav'

CPU_COUNT = multiprocessing.cpu_count()

def extract_segments(clip,frames=41):
    FRAMES_PER_SEGMENT = frames - 1  # 41 frames ~= 950 ms
    WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT  # 23 ms per frame
    STEP_SIZE = 512 * FRAMES_PER_SEGMENT // 2  # 512 * 20 = 10240
    BANDS = 60
        
    s = 0
    segments = []
    
    normalization_factor = 1 / np.max(np.abs(clip))
    clip = clip * normalization_factor
        
    logspec = 0
    if len(clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]) == WINDOW_SIZE:
        signal = clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]
        melspec = melspectrogram(signal, sr=22050, n_fft=1024, hop_length=512, n_mels=BANDS)
        logspec = amplitude_to_db(melspec)
        logspec = logspec.T.flatten()[:, np.newaxis].T
        logspec = pd.DataFrame(
        data=logspec, dtype='float32', index=[0],
        columns=list('logspec_b{}_f{}'.format(i % BANDS, i // BANDS) for i in range(np.shape(logspec)[1]))
        )

    return logspec


def generate_deltas(X):
    new_dim = np.zeros(tuple(np.shape(X)))
    X = np.concatenate((X, new_dim), axis=3)
    del new_dim
    for i in range(len(X)):
        X[i, :, :, 1] = delta(X[i, :, :, 0])
    return X

audio, sr = librosa.load(audio_path) 

a = extract_segments(audio)

shape = (-1, 60, 41, 1)
start_col = 'logspec_b0_f0'

X_train = a.values

X_mean = np.mean(X_train)
X_std = np.std(X_train)
X_train = (X_train - X_mean) / X_std

X_train = np.reshape(X_train, shape, order='F')
X_train = generate_deltas(X_train)

print(model.summary())

y_predict = model.predict(X_train)	

print(np.round(y_predict))

