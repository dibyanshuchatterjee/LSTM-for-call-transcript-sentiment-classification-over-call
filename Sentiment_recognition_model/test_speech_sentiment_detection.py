import numpy as np
import speech_sentiment_recognition as sp
from keras.models import load_model

loaded_model = load_model('speech_emotion_detector_model.h5')
class_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']


def prepare_X_test(test_files):
    X_test = []
    for file in test_files:
        mfcc = sp.extract_mfcc(file)
        X_test.append(mfcc)

    X_test = np.array(X_test)
    # input split
    X_test = np.expand_dims(X_test, -1)

    return X_test


def test():
    X_test = prepare_X_test(
        ['/Users/dibyanshuchatterjee/PycharmProjects/Call_Rep_Assistance_For_Students/recording.wav'])
    # Perform inference on test data
    predictions = loaded_model.predict(X_test)
    highest_prob_index = np.argmax(predictions)
    highest_prob_class = class_labels[highest_prob_index]
    print(highest_prob_class)


test()
