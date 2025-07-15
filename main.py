import speech_recognition as sr
import pandas as pd
from datetime import datetime
import os
from textblob import TextBlob

r = sr.Recognizer()

def record_text():
    with sr.Microphone() as source:
        print("\n🎤 Speak your complaint (say 'stop recording' to exit):")
        r.adjust_for_ambient_noise(source, duration=1)
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("✅ Complaint recorded:", text)
            return text
        except sr.UnknownValueError:
            print("❌ Could not understand audio")
            return None
        except sr.RequestError:
            print("❌ Could not request results")
            return None

def analyze_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity

def save_complaint(complaint_text):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    polarity, subjectivity = analyze_sentiment(complaint_text)
    df = pd.DataFrame([[now, complaint_text, polarity, subjectivity]],
                      columns=["Timestamp", "Complaint", "Polarity", "Subjectivity"])

    file_exists = os.path.exists("complaints.csv")
    df.to_csv("complaints.csv", mode='a', header=not file_exists, index=False)
    print("📝 Complaint saved successfully.")

# 🔁 Loop to keep recording
while True:
    complaint = record_text()
    
    if complaint:
        if "stop recording" in complaint.lower():
            print("🛑 Stopping complaint recording.")
            break
        save_complaint(complaint)
