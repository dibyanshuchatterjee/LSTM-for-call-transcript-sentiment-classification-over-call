from transformers import PegasusForConditionalGeneration
from transformers import PegasusTokenizer
from transformers import pipeline
import speech_recognition as sr


# Pick model
model_name = "google/pegasus-xsum"

# Load pretrained tokenizer
pegasus_tokenizer = PegasusTokenizer.from_pretrained(model_name)


def summarize_converted_text(text):
    # Define summarization pipeline
    summarizer = pipeline(
        "summarization",
        model=model_name,
        tokenizer=pegasus_tokenizer,
        framework="pt"
    )
    # Check if the text is too short
    if len(text.split()) < 30:
        return text
    # Create summary
    summary = summarizer(text, min_length=30, max_length=150)

    return summary[0]["summary_text"]


def convert_mp3_to_text(sound):
    r = sr.Recognizer()

    with sr.AudioFile(sound) as source:
        r.adjust_for_ambient_noise(source)

        audio = r.listen(source)

        print("Recognizing Now .... ")

        # recognize speech using google

        try:
            print("Recognized text \n" + r.recognize_google(audio))

        except Exception as e:

            print("Error :  " + str(e))
    return r.recognize_google(audio)


def main():
    text = convert_mp3_to_text(sound="recording.wav")
    summary = summarize_converted_text(text=text)
    print(summary)


if __name__ == '__main__':
    main()
