import streamlit as st
import speech_recognition as sr
from textblob import TextBlob
import pandas as pd
import os
from datetime import datetime

st.set_page_config(page_title="Voice Complaint System")

st.title("🎤 Voice to Text Complaint Logger")

def record_text():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎙 Please speak your complaint...")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            st.success(f"✅ Complaint recorded: {text}")
            return text
        except Exception as e:
            st.error(f"❌ Error: {e}")
            return None

def save_complaint(complaint_text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sentiment = TextBlob(complaint_text).sentiment
    df = pd.DataFrame([[timestamp, complaint_text, sentiment.polarity, sentiment.subjectivity]],
                      columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])
    file_exists = os.path.exists("complaints.csv")
    df.to_csv("complaints.csv", mode='a', header=not file_exists, index=False)
    st.success("📝 Complaint saved successfully.")

if st.button("🎙 Speak Now"):
    complaint = record_text()
    if complaint:
        save_complaint(complaint)

st.markdown("---")
if st.checkbox("📂 Show saved complaints"):
    if os.path.exists("complaints.csv"):
        df = pd.read_csv("complaints.csv")
        st.dataframe(df)
    else:
        st.info("No complaints saved yet.")
