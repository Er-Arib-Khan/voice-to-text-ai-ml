import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="🎤 Voice to Text AI", layout="centered")

st.title("🎤 Voice to Text Complaint App")

# Load existing data
if os.path.exists("complaints.csv"):
    df = pd.read_csv("complaints.csv")
else:
    df = pd.DataFrame(columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])

# Button to start voice recording
if st.button("🎙️ Start Recording"):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak clearly.")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)
            st.success("✅ You said: " + text)

            # Sentiment analysis
            blob = TextBlob(text)
            polarity = round(blob.sentiment.polarity, 2)
            subjectivity = round(blob.sentiment.subjectivity, 2)

            st.write(f"🧠 **Polarity:** `{polarity}`")
            st.write(f"🎯 **Subjectivity:** `{subjectivity}`")

            # Save to CSV
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            df_new = pd.DataFrame([[timestamp, text, polarity, subjectivity]],
                                  columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])
            df = pd.concat([df, df_new], ignore_index=True)
            df.to_csv("complaints.csv", index=False)
            st.success("📁 Complaint saved successfully.")

        except sr.UnknownValueError:
            st.error("❌ Sorry, could not understand.")
        except sr.RequestError:
            st.error("🔌 Could not request results. Check your internet.")
