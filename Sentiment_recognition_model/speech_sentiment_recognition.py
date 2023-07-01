import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa
import librosa.display
from IPython.display import Audio
import warnings
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

warnings.filterwarnings('ignore')


def load_dataset():
    paths = []
    labels = []
    for dirname, _, filenames in os.walk('./input'):
        for filename in filenames:
            paths.append(os.path.join(dirname, filename))
            label = filename.split('_')[-1]
            label = label.split('.')[0]
            labels.append(label.lower())
        if len(paths) == 2800:
            break
    print('Dataset is Loaded')
    print(f'Len of paths: {len(paths)}')
    print(paths[:5])

    # create df

    df = pd.DataFrame()
    df['speech'] = paths
    df['label'] = labels

    return df


def exploratory_data_analysis(df):
    # for fear
    print(sns.__version__)
    print(sns.countplot(df['label']))
    emotion = 'fear'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for anger
    emotion = 'angry'
    path = np.array(df['speech'][df['label'] == emotion])[1]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for disgust
    emotion = 'disgust'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for neutral
    emotion = 'neutral'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for sad
    emotion = 'sad'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for ps
    emotion = 'ps'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)
    # for happy
    emotion = 'happy'
    path = np.array(df['speech'][df['label'] == emotion])[0]
    data, sampling_rate = librosa.load(path)
    waveplot(data, sampling_rate, emotion)
    spectogram(data, sampling_rate, emotion)
    Audio(path)


def waveplot(data, sr, emotion):
    plt.figure(figsize=(10, 4))
    plt.title(emotion, size=20)
    librosa.display.waveplot(data, sr=sr)
    plt.show()


def spectogram(data, sr, emotion):
    x = librosa.stft(data)
    xdb = librosa.amplitude_to_db(abs(x))
    plt.figure(figsize=(11, 4))
    plt.title(emotion, size=20)
    librosa.display.specshow(xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.colorbar()


def perform_feature_extraction(df):
    X_mfcc = df['speech'].apply(lambda x: extract_mfcc(x))
    X = [x for x in X_mfcc]
    X = np.array(X)
    # input split
    X = np.expand_dims(X, -1)

    # Perform OHE
    enc = OneHotEncoder()
    y = enc.fit_transform(df[['label']])
    y = y.toarray()

    return X, y


def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc


def train_LSTM_model(X, y):
    model = Sequential([
        LSTM(256, return_sequences=False, input_shape=(40, 1)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(7, activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f'Model Summary: \n {model.summary()}')

    # Train the model
    history = model.fit(X, y, validation_split=0.2, epochs=50, batch_size=64)
    model.save('speech_emotion_detector_model.h5')


def main():
    df = load_dataset()
    # exploratory_data_analysis(df=df)
    X, y = perform_feature_extraction(df=df)
    train_LSTM_model(X=X, y=y)


if __name__ == '__main__':
    main()
