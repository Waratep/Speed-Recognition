import os

import numpy as np
import pandas as pd

import librosa
from librosa.feature import melspectrogram
from librosa.core import amplitude_to_db
from librosa.feature import delta

import multiprocessing
from joblib import Parallel, delayed

from keras.models import Sequential	
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

FILES_PATH = ''

def load_data():
  metadata = pd.read_csv(os.path.join(FILES_PATH, 'dataset.csv'))
    
  rows_meta = []
  rows_audio = []
    
  for key, row in metadata.iterrows():
    filename = row['filename']
    label = row['label']
    label_name = row['label_name']
    dataset = row['sets']
    rows_meta.append(
      pd.DataFrame({
        'filename': filename,
        'label': label, 
        'label_name': label_name,
        'sets': dataset
      }, index=[0])
    )
    audio_path = os.path.join(FILES_PATH, 'dataset', filename)
    audio, sr = librosa.load(audio_path)
    if audio is not None:
      rows_audio.append(audio)
        
  rows_meta = pd.concat(rows_meta, ignore_index=True)
  rows_audio = np.vstack(rows_audio)
  rows_meta[['label']] = rows_meta[['label']].astype(int)
    
  return rows_meta, rows_audio

def extract_segments(clip, filename, sets, label, label_name, frames):
  FRAMES_PER_SEGMENT = frames - 1  # 41 frames ~= 950 ms
  WINDOW_SIZE = 512 * FRAMES_PER_SEGMENT  # 23 ms per frame
  STEP_SIZE = 512 * FRAMES_PER_SEGMENT // 2  # 512 * 20 = 10240
  BANDS = 60
    
  s = 0
  segments = []
  
  normalization_factor = 1 / np.max(np.abs(clip))
  clip = clip * normalization_factor
    
  while len(clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]) == WINDOW_SIZE:
    signal = clip[s * STEP_SIZE:s * STEP_SIZE + WINDOW_SIZE]
    melspec = melspectrogram(signal, sr=22050, n_fft=1024, hop_length=512, n_mels=BANDS)
    logspec = amplitude_to_db(melspec)
    logspec = logspec.T.flatten()[:, np.newaxis].T
    logspec = pd.DataFrame(
      data=logspec, dtype='float32', index=[0],
      columns=list('logspec_b{}_f{}'.format(i % BANDS, i // BANDS) for i in range(np.shape(logspec)[1]))
    )
    if np.mean(logspec.values) > -70.0:
      segment_meta = pd.DataFrame({
        'filename': filename,
        'sets': sets,
        'label': label,
        'label_name': label_name,
        's_begin': s * STEP_SIZE,
        's_end': s * STEP_SIZE + WINDOW_SIZE
      }, index=[0])
      segments.append(pd.concat((segment_meta, logspec), axis=1))
            
    s = s + 1
        
  segments = pd.concat(segments, ignore_index=True)
  return segments
  
def extract_features(meta, audio, frames=41):
  segments = []
  start = 0
  end = len(audio)
  segments.extend(Parallel(n_jobs=CPU_COUNT)(delayed(extract_segments)(
    audio[i, :],
    meta.loc[i, 'filename'],
    meta.loc[i, 'sets'],
    meta.loc[i, 'label'],
    meta.loc[i, 'label_name'],
    frames) for i in range(start, end)))
  segments = pd.concat(segments, ignore_index=True)
  return segments

def generate_deltas(X):
    new_dim = np.zeros(tuple(np.shape(X)))
    X = np.concatenate((X, new_dim), axis=3)
    del new_dim

    for i in range(len(X)):
        X[i, :, :, 1] = delta(X[i, :, :, 0])
    return X
  
def to_one_hot(labels, class_count):
    one_hot_enc = np.zeros((len(labels), class_count))
    for r in range(len(labels)):
        one_hot_enc[r, labels[r]] = 1
    return one_hot_enc

if __name__ == "__main__":

    CPU_COUNT = multiprocessing.cpu_count()

    meta_data, meta_audio = load_data()

    # print('meta_data, meta_audio')
    # print(meta_audio.shape)
    # print('##################################################')

    features_data = extract_features(meta_data, meta_audio)

    # print('features_data')
    # print(features_data,features_data.shape)
    # print('##################################################')

    shape = (-1, 60, 41, 1)
    start_col = 'logspec_b0_f0'
    end_col = features_data.columns[-1]
    class_count = len(pd.unique(features_data['label']))

    train = features_data[(features_data['sets'] == 'train')]
    test = features_data[(features_data['sets'] == 'test')]

    # print(np.array(train)[0],test.shape)

    X_train = train.loc[:, start_col:end_col].values
    y_train = to_one_hot(train['label'].values, class_count)

    X_test = test.loc[:, start_col:end_col].values
    y_test = to_one_hot(test['label'].values, class_count)

    # print(np.array(y_train)[55],np.array(y_train)[0].shape)

    X_mean = np.mean(X_train)
    X_std = np.std(X_train)

    X_train = (X_train - X_mean) / X_std
    X_test = (X_test - X_mean) / X_std

    # print(X_train.shape)
    X_train = np.reshape(X_train, shape, order='F')
    X_test = np.reshape(X_test, shape, order='F')
    # print(X_train.shape)

    X_train = generate_deltas(X_train)
    X_test = generate_deltas(X_test)

    # #print(features_data.head(10))

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(60, 41, 2)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # # layer 2
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    # # full connected
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='softmax'))

    # # compile model
    model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    history = model.fit(X_train,
                    y_train,
                    batch_size=512,
                    epochs=100,
                    verbose=1,
                    validation_data=(X_test, y_test))

    # model.predict_classes()
    model.save('switch.h5')