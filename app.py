import numpy as np
import requests
from twilio.rest import Client
import summarize_calls as sum
from keras.models import load_model
import Sentiment_recognition_model.speech_sentiment_recognition as sp

loaded_model = load_model('speech_emotion_detector_model.h5')
class_labels = ['fear', 'angry', 'disgust', 'neutral', 'sad', 'ps', 'happy']


def place_call_and_record():
    # Twilio Account SID and Auth Token
    account_sid = "AC0f-----"
    auth_token = "8b-----"

    # Twilio phone number and recipient phone number
    twilio_number = "+1----"
    recipient_number = "+1----"

    # Create a Twilio client
    client = Client(account_sid, auth_token)

    # Read the TwiML from a file
    with open("twiml.xml", "r") as file:
        twiml = file.read()

    # Initiate the call and connect to your phone number
    call = client.calls.create(
        twiml=twiml,
        to=recipient_number,
        from_=twilio_number,
        record=True,
        recording_track='inbound'
    )

    print("Call started. SID:", call.sid)

    # Wait for the call to end
    input("Press Enter when the call is complete...")

    # Retrieve the call details
    call = client.calls(call.sid).fetch()

    # Retrieve the recording SID from the list of recordings
    recordings = client.recordings.list(call_sid=call.sid)
    print(recordings)
    if len(recordings) > 0:
        recording_sid = recordings[0].sid

        # Retrieve the recording details
        recording = client.recordings(recording_sid).fetch()

        # Get the recording URL
        # Get the URL of the recording
        recording_url = 'https://api.twilio.com' + recording.uri.replace('.json', '.wav')

        # Download the recording
        # response = client.request("GET", recording_url + ".mp3")
        response = requests.get(recording_url, auth=(account_sid, auth_token), stream=True)

        # Save the recording to a file
        with open("recording.wav", "wb") as f:
            f.write(response.content)

        print("Recording downloaded.")
    else:
        print("No recordings found for the call.")


def prepare_X_test(test_files):
    X_test = []
    for file in test_files:
        mfcc = sp.extract_mfcc(file)
        X_test.append(mfcc)

    X_test = np.array(X_test)
    # input split
    X_test = np.expand_dims(X_test, -1)

    return X_test


def predict_class(sound):
    X_test = prepare_X_test(sound)
    # Perform inference on test data
    predictions = loaded_model.predict(X_test)
    highest_prob_index = np.argmax(predictions)
    highest_prob_class = class_labels[highest_prob_index]
    return highest_prob_class


def main():
    place_call_and_record()
    text = sum.convert_mp3_to_text(sound='recording.wav')
    summary = sum.summarize_converted_text(text=text)
    emotion = predict_class(sound=['/Users/dibyanshuchatterjee/PycharmProjects/Call_Rep_Assistance_For_Students'
                                   '/recording.wav'])

    print(summary)
    print(emotion)


if __name__ == '__main__':
    main()
