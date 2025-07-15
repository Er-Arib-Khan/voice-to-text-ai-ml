import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="🎙️ Voice to Text AI", layout="centered")
st.title("🎤 Voice to Text Complaint App")

r = sr.Recognizer()

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    st.info("Processing your uploaded audio...")

    # Save the uploaded file
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        with sr.AudioFile("temp.wav") as source:
            audio = r.record(source)
            complaint_text = r.recognize_google(audio)

            st.success("📝 Transcription: " + complaint_text)

            # Sentiment analysis
            blob = TextBlob(complaint_text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Save to CSV
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df = pd.DataFrame([[now, complaint_text, polarity, subjectivity]],
                            columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])
            file_exists = os.path.exists("complaints.csv")
            df.to_csv("complaints.csv", mode='a', header=not file_exists, index=False)

            st.success("✅ Complaint saved to CSV")

    except sr.UnknownValueError:
        st.error("Sorry, could not understand the audio.")
    except sr.RequestError:
        st.error("Could not request results. Check your internet connection.")
else:
    st.info("📢 Please upload a WAV file to proceed.")
